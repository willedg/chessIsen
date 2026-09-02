// wp3/mcts.cpp - Optimized MCTS with Arena Allocation and Tree Reuse support
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <unordered_map>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <chrono>
#include <memory>

#include "board_encoder.hpp"

namespace py = pybind11;

// Structure de nœud compacte pour minimiser l'empreinte mémoire
struct Node {
    std::unordered_map<int, uint32_t> children; // Utilise des index au lieu de pointeurs pour l'Aréna
    int parent_idx = -1;
    int move_idx = -1;
    float prior = 0.0f;
    float value_sum = 0.0f;
    int visit_count = 0;
    int virtual_loss = 0;
    bool is_expanded = false;
    bool is_terminal = false;
    float terminal_val = 0.0f;

    void reset(float p = 0.0f, int p_idx = -1, int mv_idx = -1) {
        children.clear();
        parent_idx = p_idx;
        move_idx = mv_idx;
        prior = p;
        value_sum = 0.0f;
        visit_count = 0;
        virtual_loss = 0;
        is_expanded = false;
        is_terminal = false;
        terminal_val = 0.0f;
    }
};

// Aréna de nœuds pour éviter les allocations/désallocations répétées
class NodeArena {
    std::vector<Node> pool;
    uint32_t next_free = 0;
public:
    NodeArena(size_t initial_cap = 1000000) {
        pool.resize(initial_cap);
    }
    
    uint32_t allocate(float prior = 0.0f, int parent_idx = -1, int move_idx = -1) {
        if (next_free >= pool.size()) {
            pool.resize(pool.size() * 1.5);
        }
        uint32_t idx = next_free++;
        pool[idx].reset(prior, parent_idx, move_idx);
        return idx;
    }

    Node& operator[](uint32_t idx) { return pool[idx]; }
    void clear() { next_free = 0; }
    uint32_t size() const { return next_free; }
    
    // Garde une partie des nœuds (Tree Reuse)
    void reuse_subtree(uint32_t new_root_idx) {
        if (new_root_idx == 0) return;
        // Pour simplifier le Tree Reuse robuste, on va juste copier le sous-arbre 
        // ou marquer les anciens nœuds comme libres.
        // Version simple pour l'instant : on reset tout sauf si on implémente un ramasse-miettes.
        // On va juste repartir de 0 si on n'a pas un vrai Tree Reuse complexe.
        next_free = 0; 
    }
};

class MCTS {
public:
    MCTS(py::function predict_fn, double c_puct = 1.25, 
         int batch_size = 64, int top_k = 10000, double dir_eps = 0.25, double dir_alpha = 0.03, int seed = -1)
        : predict(predict_fn), c_puct((float)c_puct), batch_size(batch_size), top_k(top_k), 
          dirichlet_eps((float)dir_eps), dirichlet_alpha((float)dir_alpha), arena(500000)
    {
        root_idx = arena.allocate(1.0f, -1, -1);
        if (seed >= 0) {
            rng.seed(seed);
        } else {
            std::random_device rd;
            rng.seed(rd());
        }
    }

    py::tuple search(py::object board, int n_sim, float temperature = 1.0f, float noise_eps = -1.0f) {
        // Init board pool
        if (board_pool.empty()) {
            for (int i = 0; i < batch_size; ++i) {
                py::object b = board.attr("copy")();
                board_pool.push_back(b);
                push_funcs.push_back(b.attr("push"));
                pop_funcs.push_back(b.attr("pop"));
                reset_funcs.push_back(b.attr("reset"));
            }
        }

        float current_eps = (noise_eps >= 0.0f) ? noise_eps : this->dirichlet_eps;

        // --- NOUVEAU: Force Root Expansion ---
        // Si la racine n'est pas développée, on le fait AVANT la boucle
        if (!arena[root_idx].is_expanded) {
            py::list root_list;
            root_list.append(board);
            py::list outputs = predict(root_list);
            py::tuple out = outputs[0].cast<py::tuple>();
            
            if (move_objects.empty()) {
                move_objects = out[3].cast<std::vector<py::object>>();
                move_ucis = out[4].cast<std::vector<std::string>>();
            }

            py::array_t<float> logits = out[0].cast<py::array_t<float>>();
            std::vector<int> legal_indices = out[2].cast<std::vector<int>>();
            
            if (legal_indices.empty()) {
                arena[root_idx].is_terminal = true;
                // ... (gestion du résultat)
            } else {
                expand_node_from_logits(root_idx, logits, legal_indices);
            }
        }

        // Appliquer le bruit Dirichlet à la racine
        if (!noise_applied && current_eps > 0.0001f && arena[root_idx].is_expanded) {
            apply_dirichlet_noise(root_idx, current_eps);
            noise_applied = true;
        }

        auto start = std::chrono::high_resolution_clock::now();
        int sims_done = 0;
        
        while (sims_done < n_sim) {
            int cur_batch = std::min(batch_size, n_sim - sims_done);
            std::vector<uint32_t> leaves;
            std::vector<std::vector<uint32_t>> paths(cur_batch);
            std::vector<int> leaf_indices;

            for (int i = 0; i < cur_batch; ++i) {
                uint32_t node_idx = root_idx;
                paths[i].push_back(node_idx);
                
                while (arena[node_idx].is_expanded && !arena[node_idx].is_terminal) {
                    float best_s = -1e30f; uint32_t best_c_idx = 0; int best_mv_idx = -1;
                    float u_fact = c_puct * std::sqrt((float)arena[node_idx].visit_count + (float)arena[node_idx].virtual_loss + 1e-8f);
                    
                    for (auto& kv : arena[node_idx].children) {
                        Node& child = arena[kv.second];
                        float visits = (float)child.visit_count + (float)child.virtual_loss;
                        float q = (visits > 0.1f) ? (-child.value_sum - (float)child.virtual_loss) / visits : 0.0f;
                        
                        // Clamp Q
                        if (q > 1.0f) q = 1.0f; else if (q < -1.0f) q = -1.0f;
                        
                        float s = q + u_fact * child.prior / (1.0f + visits);
                        if (s > best_s) { best_s = s; best_c_idx = kv.second; best_mv_idx = kv.first; }
                    }
                    if (best_mv_idx == -1) break;
                    
                    arena[best_c_idx].virtual_loss++;
                    paths[i].push_back(best_c_idx);
                    push_funcs[i](move_objects[best_mv_idx]); 
                    node_idx = best_c_idx;
                }
                
                if (arena[node_idx].is_terminal) {
                    backprop(paths[i], arena[node_idx].terminal_val);
                    for (size_t k = 1; k < paths[i].size(); ++k) {
                        pop_funcs[i]();
                        if (arena[paths[i][k]].virtual_loss > 0) arena[paths[i][k]].virtual_loss--;
                    }
                } else {
                    leaves.push_back(node_idx);
                    leaf_indices.push_back(i);
                }
            }

            if (!leaves.empty()) {
                py::list batch_boards;
                for (int idx : leaf_indices) batch_boards.append(board_pool[idx]);
                py::list outputs = predict(batch_boards);
                
                for (size_t j = 0; j < leaves.size(); ++j) {
                    int i = leaf_indices[j]; uint32_t leaf_idx = leaves[j];
                    py::tuple out = outputs[j].cast<py::tuple>();
                    
                    if (move_objects.empty()) {
                        move_objects = out[3].cast<std::vector<py::object>>();
                        move_ucis = out[4].cast<std::vector<std::string>>();
                    }

                    py::array_t<float> logits = out[0].cast<py::array_t<float>>();
                    float v = out[1].cast<float>();
                    std::vector<int> legal_indices = out[2].cast<std::vector<int>>();

                    if (!arena[leaf_idx].is_expanded) {
                        if (legal_indices.empty()) {
                            arena[leaf_idx].is_terminal = true;
                            std::string res = board_pool[i].attr("result")().cast<std::string>();
                            bool white_to_move = board_pool[i].attr("turn").cast<bool>();
                            float reward = 0.0f;
                            if (res == "1-0") reward = white_to_move ? 1.0f : -1.0f;
                            else if (res == "0-1") reward = white_to_move ? -1.0f : 1.0f;
                            arena[leaf_idx].terminal_val = reward;
                            v = reward;
                        } else {
                            expand_node_from_logits(leaf_idx, logits, legal_indices);
                        }
                    }
                    backprop(paths[i], v);
                    for (size_t k = 1; k < paths[i].size(); ++k) {
                        pop_funcs[i]();
                        if (arena[paths[i][k]].virtual_loss > 0) arena[paths[i][k]].virtual_loss--;
                    }
                }
            }
            sims_done += cur_batch;

            // --- Optimisation Early Exit ---
            // Si on a déjà assez de simulations et qu'un coup domine clairement, on s'arrête.
            if (sims_done >= 400 && sims_done % batch_size == 0) {
                int max_v = 0;
                int second_max_v = 0;
                for (auto& kv : arena[root_idx].children) {
                    int v = arena[kv.second].visit_count;
                    if (v > max_v) {
                        second_max_v = max_v;
                        max_v = v;
                    } else if (v > second_max_v) {
                        second_max_v = v;
                    }
                }
                // Si le meilleur coup a déjà 3x plus de visites que le 2ème, on sort.
                if (max_v > second_max_v * 3 && second_max_v > 0) {
                    break;
                }
            }
        }
        return collect_results(board, n_sim, start);
    }

    void advance_root(int idx, py::object move_obj) {
        if (idx == -1) {
            arena.clear();
            root_idx = arena.allocate(1.0f, -1, -1);
            for (auto& reset_fn : reset_funcs) reset_fn();
            noise_applied = false;
            return;
        }

        auto it = arena[root_idx].children.find(idx);
        if (it != arena[root_idx].children.end()) { 
            uint32_t new_root_idx = it->second;
            // Pour l'instant on garde l'arène telle quelle mais on change juste la racine
            // car supprimer les nœuds orphelins dans une arène est complexe.
            // On se contente de réinitialiser périodiquement si l'arène est trop pleine.
            root_idx = new_root_idx;
        } else { 
            arena.clear(); 
            root_idx = arena.allocate(1.0f, -1, -1); 
        }
        
        if (!board_pool.empty()) {
            for (auto& push_fn : push_funcs) push_fn(move_obj);
        }
        noise_applied = false;

        // Sécurité : si on a trop de nœuds, on vide tout pour éviter OOM
        if (arena.size() > 800000) {
            arena.clear();
            root_idx = arena.allocate(1.0f, -1, -1);
        }
    }

private:
    py::function predict;
    uint32_t root_idx;
    float c_puct;
    int batch_size, top_k;
    float dirichlet_eps, dirichlet_alpha;
    bool noise_applied = false;
    std::mt19937 rng;
    NodeArena arena;
    
    std::vector<py::object> board_pool;
    std::vector<py::function> push_funcs;
    std::vector<py::function> pop_funcs;
    std::vector<py::function> reset_funcs;
    std::vector<py::object> move_objects;
    std::vector<std::string> move_ucis;

    void expand_node_from_logits(uint32_t node_idx, py::array_t<float> logits, const std::vector<int>& legal_indices) {
        auto r = logits.unchecked<1>();
        std::vector<float> f_probs;
        float max_l = -1e30f;
        for (int idx : legal_indices) if (r(idx) > max_l) max_l = r(idx);
        float sum = 0.0f;
        for (int idx : legal_indices) {
            float p = std::exp(std::min(r(idx) - max_l, 20.0f));
            f_probs.push_back(p);
            sum += p;
        }
        for (size_t i = 0; i < legal_indices.size(); ++i) {
            float p = f_probs[i] / (sum + 1e-12f);
            arena[node_idx].children[legal_indices[i]] = arena.allocate(p, node_idx, legal_indices[i]);
        }
        arena[node_idx].is_expanded = true;
    }

    void apply_dirichlet_noise(uint32_t n_idx, float eps) {
        if (arena[n_idx].children.empty() || eps <= 0) return;
        std::gamma_distribution<float> dist(dirichlet_alpha, 1.0f);
        std::vector<float> noise; float s = 0;
        for (size_t k = 0; k < arena[n_idx].children.size(); ++k) {
            float v = dist(rng); noise.push_back(v); s += v;
        }
        int k = 0;
        for (auto& kv : arena[n_idx].children) {
            float nv = noise[k++] / (s + 1e-10f);
            arena[kv.second].prior = (1.0f - eps) * arena[kv.second].prior + eps * nv;
        }
    }

    void backprop(std::vector<uint32_t>& p, float v) {
        float cv = v;
        for (auto it = p.rbegin(); it != p.rend(); ++it) {
            arena[*it].visit_count++;
            arena[*it].value_sum += cv;
            cv = -cv;
        }
    }

    py::tuple collect_results(py::object board, int n_sim, std::chrono::steady_clock::time_point start) {
        auto pi_vec = py::array_t<float>(4672);
        auto r = pi_vec.mutable_unchecked<1>();
        for (int i = 0; i < 4672; ++i) r(i) = 0.0f;

        double total_v = 0.0;
        for (auto& kv : arena[root_idx].children) total_v += (double)arena[kv.second].visit_count;
        if (total_v < 1e-12) total_v = 1e-12;

        std::string bm_uci = ""; int max_v = -1;
        for (auto& kv : arena[root_idx].children) {
            float p = (float)((double)arena[kv.second].visit_count / total_v);
            r(kv.first) = p;
            if (arena[kv.second].visit_count > max_v) { 
                max_v = arena[kv.second].visit_count; 
                bm_uci = move_ucis[kv.first]; 
            }
        }
        auto d = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
        py::dict stats;
        stats["nps"] = (d > 0) ? (double)n_sim * 1000.0 / d : 0.0;
        stats["nodes"] = arena.size();
        return py::make_tuple(pi_vec, bm_uci, stats);
    }
};

PYBIND11_MODULE(cpp_mcts, m) {
    m.def("encode_batch", &BoardEncoder::encode_batch, "Fast C++ board encoding");
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<py::function,double,int,int,double,double,int>(),
             py::arg("predict_fn"), py::arg("c_puct") = 1.25, py::arg("batch_size") = 64, py::arg("top_k") = 10000,
             py::arg("dirichlet_eps") = 0.25, py::arg("dirichlet_alpha") = 0.03, py::arg("seed") = -1
        )
        .def("search", &MCTS::search,
             py::arg("board"), py::arg("n_sim"), 
             py::arg("temperature") = 1.0f, py::arg("noise_eps") = -1.0f)
        .def("advance_root", &MCTS::advance_root);
}
