#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <string>
#include <vector>

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_debug_cb_user_data cb_data;

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = [](struct ggml_tensor * t, bool ask, void * user_data) -> bool {
        bool ret = common_debug_cb_eval(t, ask, user_data);
        if (!ask && t->op == GGML_OP_MUL_MAT) {
            if (strcmp(t->name, "ffn_out-15") == 0) {
                static bool first_match = true;
                if (first_match) {
                    first_match = false;
                    fprintf(stderr, "Found blk.15.ffn_down (tensor: %s) for the first time, dumping tensors...\n", t->name);
                    
                    auto write_tensor = [](struct ggml_tensor * tensor, const char * filename) {
                        int64_t nelements = ggml_nelements(tensor);
                        size_t nbytes = ggml_nbytes(tensor);
                        std::vector<char> buffer(nbytes);
                        ggml_backend_tensor_get(tensor, buffer.data(), 0, nbytes);
                        
                        std::vector<float> f32_data;
                        void * data_ptr = buffer.data();
                        
                        if (tensor->type == GGML_TYPE_F16) {
                            f32_data.resize(nelements);
                            ggml_fp16_t * f16_ptr = (ggml_fp16_t *) buffer.data();
                            for (int64_t i = 0; i < nelements; ++i) {
                                f32_data[i] = ggml_fp16_to_fp32(f16_ptr[i]);
                            }
                            data_ptr = f32_data.data();
                            nbytes = nelements * sizeof(float);
                        } else if (tensor->type != GGML_TYPE_F32) {
                            fprintf(stderr, "Warning: tensor %s is not F32 or F16, type=%d\n", tensor->name, tensor->type);
                        }
                        
                        FILE * f = fopen(filename, "wb");
                        if (f) {
                            fwrite(data_ptr, 1, nbytes, f);
                            fclose(f);
                            fprintf(stderr, "Dumped %s to %s: %ld elements (%zu bytes), shape=[%ld, %ld, %ld, %ld]\n",
                                tensor->name, filename, nelements, nbytes, tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
                        } else {
                            fprintf(stderr, "Failed to open %s for writing\n", filename);
                        }
                    };

                    struct ggml_tensor * src1 = t->src[1];
                    if (src1) {
                        write_tensor(src1, "x_blk15_ffn_down_input.bin");
                    }
                    write_tensor(t, "y_blk15_ffn_down_output.bin");
                }
            }
        }
        return ret;
    };
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
