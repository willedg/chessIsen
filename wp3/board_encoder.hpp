// wp3/board_encoder.hpp - Ultra High Performance Bitboard Encoder
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>

namespace py = pybind11;

class BoardEncoder {
public:
    static py::array_t<float> encode_batch(py::list boards) {
        size_t batch_size = py::len(boards);
        std::vector<std::intptr_t> shape = {(std::intptr_t)batch_size, 19, 8, 8};
        auto result = py::array_t<float>(shape);
        float* ptr = (float*)result.request().ptr;
        std::fill(ptr, ptr + batch_size * 1216, 0.0f);
        for (size_t i = 0; i < batch_size; ++i) encode_single(boards[i], ptr + i * 1216);
        return result;
    }

    static void encode_single(py::object board, float* planes) {
        static py::function fast_info_fn = py::module::import("wp2.encoders").attr("get_fast_state_info");
        
        py::tuple info = fast_info_fn(board).cast<py::tuple>();

        // 12 piece bitboards
        for (int p = 0; p < 12; ++p) {
            uint64_t bb = info[p].cast<uint64_t>();
            while (bb) {
                int sq = trailing_zeros(bb);
                int rank = 7 - (sq / 8);
                int file = sq % 8;
                planes[p * 64 + rank * 8 + file] = 1.0f;
                bb &= bb - 1;
            }
        }

        // turn
        float turn_val = info[12].cast<bool>() ? 1.0f : 0.0f;
        for (int j = 0; j < 64; ++j) planes[12 * 64 + j] = turn_val;

        // castling
        for (int p = 13; p <= 16; ++p) {
            float val = info[p].cast<bool>() ? 1.0f : 0.0f;
            for (int s = 0; s < 64; ++s) planes[p * 64 + s] = val;
        }

        // move clock
        float halfmove = (float)info[17].cast<int>() / 100.0f;
        for (int j = 0; j < 64; ++j) planes[17 * 64 + j] = halfmove;

        // repetition
        float rep_val = info[18].cast<bool>() ? 1.0f : 0.0f;
        for (int j = 0; j < 64; ++j) planes[18 * 64 + j] = rep_val;
    }

private:
    static inline int trailing_zeros(uint64_t bb) {
#ifdef _MSC_VER
        unsigned long index;
        if (_BitScanForward64(&index, bb)) return (int)index;
        return 64;
#else
        if (bb == 0) return 64;
        return __builtin_ctzll(bb);
#endif
    }
};
