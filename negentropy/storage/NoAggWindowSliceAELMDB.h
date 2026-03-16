#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "lmdbxx/lmdb++.h"
#include "negentropy.h"

#ifndef MDB_AELMDB_VERSION
#error "LMDB header is not the AELMDB fork."
#endif

namespace negentropy { namespace storage {

/**
 * NoAggWindowSliceAELMDB: Negentropy StorageBase backend over AELMDB basic
 * aggregate APIs, intentionally without the advanced cached-window helpers.
 *
 * This implementation is a benchmark/storage counterpart to SliceAELMDB:
 * it preserves the same TSID key interpretation, slice semantics, and public
 * interface, but computes slice mapping and range aggregates using only:
 *   - dbi.totals()
 *   - dbi.rank(..., lmdb::agg_weight::entries, lmdb::agg_rank_mode::set_range, ...)
 *   - dbi.select(...)
 *   - cursor.seek_rank(abs_rank)
 *   - dbi.range(...)
 *
 * It deliberately does NOT use:
 *   - MDB_agg_window
 *   - dbi.window_aggregate(...)
 *   - dbi.window_rank(...)
 *
 * Required DBI schema (checked at runtime in the constructor):
 *   - MDB_AGG_ENTRIES
 *   - MDB_AGG_HASHSUM
 *   - MDB_AGG_HASHSOURCE_FROM_KEY
 *
 * Hash offset / key layout:
 *   For the (timestamp,id) schema used by Negentropy, we assume that:
 *     - the record key begins with an 8-byte big-endian timestamp
 *     - the Item id bytes begin at the per-DB hash offset (md_hash_offset)
 *     - MDB_HASH_SIZE == negentropy::ID_SIZE
 *
 * NOTE: If md_hash_offset > 8, the Bound-based constructor fills the bytes
 * between the timestamp and id with zeros, exactly like SliceAELMDB.
 * If the key schema uses non-zero bytes there, use the raw-key constructor
 * and pass full boundary keys.
 */

static_assert(MDB_HASH_SIZE == negentropy::ID_SIZE,
              "NoAggWindowSliceAELMDB expects MDB_HASH_SIZE == negentropy::ID_SIZE");

class NoAggWindowAELMDBKeyCodecTSID {
public:
    static constexpr std::size_t TS_WIRE_SIZE = 8;
    static constexpr std::size_t ID_SIZE_     = negentropy::ID_SIZE;

    explicit NoAggWindowAELMDBKeyCodecTSID(std::size_t id_offset)
        : id_offset_(id_offset) {
        if (id_offset_ < TS_WIRE_SIZE) {
            throw negentropy::err("NoAggWindowSliceAELMDB: md_hash_offset < 8 (invalid for TSID keys)");
        }
    }

    std::size_t id_offset() const noexcept { return id_offset_; }
    std::size_t key_size() const noexcept { return id_offset_ + ID_SIZE_; }

    static uint64_t load_be64(const uint8_t* p) noexcept {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v = (v << 8) | uint64_t(p[i]);
        return v;
    }

    static void store_be64(uint8_t* p, uint64_t v) noexcept {
        for (int i = 7; i >= 0; --i) { p[i] = uint8_t(v & 0xffu); v >>= 8; }
    }

    std::string encode_bound_min_key(const negentropy::Bound& b) const {
        std::string out(key_size(), '\0');
        store_be64(reinterpret_cast<uint8_t*>(out.data()), b.item.timestamp);
        std::memcpy(out.data() + id_offset_,
                    b.item.id,
                    std::min<std::size_t>(b.idLen, ID_SIZE_));
        return out;
    }

    std::string copy_key_bytes(std::string_view raw_key) const {
        if (raw_key.size() != key_size()) {
            throw negentropy::err("NoAggWindowSliceAELMDB: bad raw key size (does not match DB schema)");
        }
        return std::string(raw_key.data(), raw_key.size());
    }

    negentropy::Item decode_item_from_key(std::string_view k) const {
        if (k.size() < key_size()) {
            throw negentropy::err("NoAggWindowSliceAELMDB: key too small");
        }
        const auto* p = reinterpret_cast<const uint8_t*>(k.data());
        negentropy::Item it(load_be64(p));
        std::memcpy(it.id, p + id_offset_, ID_SIZE_);
        return it;
    }

private:
    std::size_t id_offset_;
};

struct NoAggWindowSliceAELMDB : negentropy::StorageBase {
    lmdb::txn& txn;
    const lmdb::dbi& dbi;

    unsigned int agg_flags_ = 0;
    unsigned int hash_offset_ = 0;

    NoAggWindowAELMDBKeyCodecTSID codec_;

    std::optional<std::string> begin_key_;
    std::optional<std::string> end_key_;

    mutable std::optional<lmdb::cursor> cur_;

    uint64_t total_entries_ = 0;
    uint64_t abs_lo_ = 0;
    uint64_t abs_hi_ = 0;
    uint64_t slice_size_ = 0;

    mutable negentropy::Item scratch_{};

    NoAggWindowSliceAELMDB(lmdb::txn& txn_,
                           const lmdb::dbi& dbi_,
                           std::optional<std::string_view> begin_raw_key = std::nullopt,
                           std::optional<std::string_view> end_raw_key   = std::nullopt)
        : txn(txn_),
          dbi(dbi_),
          codec_(init_schema_and_get_id_offset_()) {

        if (begin_raw_key) begin_key_ = codec_.copy_key_bytes(*begin_raw_key);
        if (end_raw_key)   end_key_   = codec_.copy_key_bytes(*end_raw_key);
        init_abs_slice_();
    }

    NoAggWindowSliceAELMDB(lmdb::txn& txn_,
                           const lmdb::dbi& dbi_,
                           const negentropy::Bound& lower,
                           const negentropy::Bound& upper)
        : txn(txn_),
          dbi(dbi_),
          codec_(init_schema_and_get_id_offset_()) {

        if (!(lower == negentropy::Bound(0))) {
            begin_key_ = codec_.encode_bound_min_key(lower);
        }
        if (!(upper == negentropy::Bound(negentropy::MAX_U64))) {
            end_key_ = codec_.encode_bound_min_key(upper);
        }
        init_abs_slice_();
    }

    uint64_t size() override { return slice_size_; }

    const negentropy::Item& getItem(size_t i) override {
        if (i >= slice_size_) {
            throw negentropy::err("NoAggWindowSliceAELMDB: bad index");
        }
        scratch_ = item_at_abs_rank_(abs_lo_ + uint64_t(i));
        return scratch_;
    }

    void iterate(size_t begin,
                 size_t end,
                 std::function<bool(const negentropy::Item&, size_t)> cb) override {
        check_bounds_(begin, end);
        if (begin == end) return;

        const uint64_t abs_begin = abs_lo_ + uint64_t(begin);
        const uint64_t abs_end   = abs_lo_ + uint64_t(end);

        auto& cur = cursor_();
        std::string_view k{}, v{};
        if (!cur.seek_rank(abs_begin, k, v)) {
            throw negentropy::err("NoAggWindowSliceAELMDB: cursor seek_rank(abs_begin) failed");
        }

        uint64_t abs_i = abs_begin;
        while (abs_i < abs_end) {
            scratch_ = codec_.decode_item_from_key(k);
            const size_t rel_i = size_t(abs_i - abs_lo_);

            if (!cb(scratch_, rel_i)) break;

            ++abs_i;
            if (abs_i >= abs_end) break;
            if (!cur.get(k, v, MDB_NEXT)) break;
        }
    }

    size_t findLowerBound(size_t begin,
                          size_t end,
                          const negentropy::Bound& value) override {
        check_bounds_(begin, end);

        const std::string bound_key = codec_.encode_bound_min_key(value);
        uint64_t abs_candidate = rank_lower_bound_(bound_key);

        const uint64_t abs_begin = abs_lo_ + uint64_t(begin);
        const uint64_t abs_end   = abs_lo_ + uint64_t(end);

        abs_candidate = std::min(std::max(abs_candidate, abs_begin), abs_end);
        return size_t(abs_candidate - abs_lo_);
    }

    negentropy::Fingerprint fingerprint(size_t begin, size_t end) override {
        check_bounds_(begin, end);

        const uint64_t n = uint64_t(end - begin);
        negentropy::Accumulator acc;
        acc.setToZero();

        if (n == 0) return acc.getFingerprint(0);

        const uint64_t abs_begin = abs_lo_ + uint64_t(begin);
        const uint64_t abs_end   = abs_lo_ + uint64_t(end);

        std::optional<std::string_view> low_key_sv;
        std::optional<std::string_view> high_key_sv;
        std::optional<std::string> low_key_owned;
        std::optional<std::string> high_key_owned;
        std::string_view tmp_k{}, tmp_v{};

        if (begin == 0) {
            if (begin_key_) {
                low_key_sv = std::string_view(begin_key_->data(), begin_key_->size());
            } else if (abs_begin == 0) {
                low_key_sv.reset();
            } else {
                if (!dbi.select(txn, lmdb::agg_weight::entries, abs_begin, tmp_k, tmp_v)) {
                    throw negentropy::err("NoAggWindowSliceAELMDB: select(abs_begin) failed in fingerprint");
                }
                low_key_owned.emplace(tmp_k.data(), tmp_k.size());
                low_key_sv = std::string_view(low_key_owned->data(), low_key_owned->size());
            }
        } else {
            if (!dbi.select(txn, lmdb::agg_weight::entries, abs_begin, tmp_k, tmp_v)) {
                throw negentropy::err("NoAggWindowSliceAELMDB: select(abs_begin) failed in fingerprint");
            }
            low_key_owned.emplace(tmp_k.data(), tmp_k.size());
            low_key_sv = std::string_view(low_key_owned->data(), low_key_owned->size());
        }

        std::string_view tmp_k2{}, tmp_v2{};
        if (end == slice_size_) {
            if (end_key_) {
                high_key_sv = std::string_view(end_key_->data(), end_key_->size());
            } else if (abs_end == total_entries_) {
                high_key_sv.reset();
            } else {
                if (!dbi.select(txn, lmdb::agg_weight::entries, abs_end, tmp_k2, tmp_v2)) {
                    throw negentropy::err("NoAggWindowSliceAELMDB: select(abs_end) failed in fingerprint");
                }
                high_key_owned.emplace(tmp_k2.data(), tmp_k2.size());
                high_key_sv = std::string_view(high_key_owned->data(), high_key_owned->size());
            }
        } else {
            if (!dbi.select(txn, lmdb::agg_weight::entries, abs_end, tmp_k2, tmp_v2)) {
                throw negentropy::err("NoAggWindowSliceAELMDB: select(abs_end) failed in fingerprint");
            }
            high_key_owned.emplace(tmp_k2.data(), tmp_k2.size());
            high_key_sv = std::string_view(high_key_owned->data(), high_key_owned->size());
        }

        const unsigned int flags = MDB_RANGE_LOWER_INCL;
        const std::string_view* lowp  = low_key_sv  ? &*low_key_sv  : nullptr;
        const std::string_view* highp = high_key_sv ? &*high_key_sv : nullptr;
        lmdb::agg a = dbi.range(txn, lowp, nullptr, highp, nullptr, flags);

        if (!a.has_hashsum() || !a.has_entries()) {
            throw negentropy::err("NoAggWindowSliceAELMDB: DBI missing HASHSUM/ENTRIES");
        }
        if (!a.has_hashsource_from_key()) {
            throw negentropy::err("NoAggWindowSliceAELMDB: DBI must hash from key (MDB_AGG_HASHSOURCE_FROM_KEY)");
        }
        if (a.mv_agg_entries != n) {
            throw negentropy::err("NoAggWindowSliceAELMDB: fingerprint range count mismatch");
        }

        std::memcpy(acc.buf, a.hashsum_data(), negentropy::ID_SIZE);
        return acc.getFingerprint(n);
    }

private:
    void check_bounds_(size_t begin, size_t end) const {
        if (begin > end || end > slice_size_) {
            throw negentropy::err("NoAggWindowSliceAELMDB: bad range");
        }
    }

    lmdb::cursor& cursor_() const {
        if (!cur_) cur_.emplace(lmdb::cursor::open(txn, dbi));
        return *cur_;
    }

    std::size_t init_schema_and_get_id_offset_() {
#if !defined(MDB_AGG_MASK)
        throw negentropy::err("NoAggWindowSliceAELMDB requires AELMDB aggregate support (MDB_AGG_MASK)");
#else
        agg_flags_ = dbi.agg_flags(txn);
        if ((agg_flags_ & MDB_AGG_ENTRIES) == 0) {
            throw negentropy::err("NoAggWindowSliceAELMDB: DBI missing MDB_AGG_ENTRIES");
        }
        if ((agg_flags_ & MDB_AGG_HASHSUM) == 0) {
            throw negentropy::err("NoAggWindowSliceAELMDB: DBI missing MDB_AGG_HASHSUM");
        }
#ifdef MDB_AGG_HASHSOURCE_FROM_KEY
        if ((agg_flags_ & MDB_AGG_HASHSOURCE_FROM_KEY) == 0) {
            throw negentropy::err("NoAggWindowSliceAELMDB: DBI missing MDB_AGG_HASHSOURCE_FROM_KEY");
        }
#endif

        hash_offset_ = dbi.hash_offset(txn);
        return std::size_t(hash_offset_);
#endif
    }

    void init_abs_slice_() {
        lmdb::agg totals = dbi.totals(txn);
        if (!totals.has_entries()) {
            throw negentropy::err("NoAggWindowSliceAELMDB: DBI missing MDB_AGG_ENTRIES");
        }

        total_entries_ = totals.mv_agg_entries;
        abs_lo_ = begin_key_ ? rank_lower_bound_(*begin_key_) : 0;
        abs_hi_ = end_key_   ? rank_lower_bound_(*end_key_)   : total_entries_;

        if (abs_hi_ < abs_lo_) abs_hi_ = abs_lo_;
        slice_size_ = abs_hi_ - abs_lo_;
    }

    uint64_t rank_lower_bound_(const std::string& key_bytes) const {
        std::string_view k{key_bytes.data(), key_bytes.size()};
        std::string_view d{};
        uint64_t r = 0;

        const bool found = dbi.rank(txn,
                                    k,
                                    d,
                                    lmdb::agg_weight::entries,
                                    lmdb::agg_rank_mode::set_range,
                                    r,
                                    nullptr);
        return found ? r : total_entries_;
    }

    negentropy::Item item_at_abs_rank_(uint64_t abs_rank) const {
        auto& cur = cursor_();
        std::string_view k{}, v{};
        if (!cur.seek_rank(abs_rank, k, v)) {
            throw negentropy::err("NoAggWindowSliceAELMDB: seek_rank(abs_rank) out of range");
        }
        return codec_.decode_item_from_key(k);
    }
};

struct NoAggWindowWholeAELMDB : NoAggWindowSliceAELMDB {
    NoAggWindowWholeAELMDB(lmdb::txn& txn, const lmdb::dbi& dbi)
        : NoAggWindowSliceAELMDB(txn, dbi, std::nullopt, std::nullopt) {}
};

}} // namespace negentropy::storage