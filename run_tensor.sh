# CUDA_VISIBLE_DEVICES=1 python -u tensor_train.py -tensorized 1 -uncompressed 0 -precondition 0 -n_warmup_steps 50000 -data data/multi30k.atok.low.pt -save_model model/test_ATIS_tensor_2layer_INT4 -save_mode best -proj_share_weight -label_smoothing -batch_size 64 -epoch 500|tee logs/test_ATIS_tensor_2layer_INT4.txt

file=ATIS_notensor_2layers_FP32_99
CUDA_VISIBLE_DEVICES=2 python -u tensor_train.py -tensorized 0 -uncompressed 0 -precondition 0 -save_model model/${file} -save_mode best -proj_share_weight -label_smoothing -batch_size 32 -epoch 500 -config configs/ATIS/GraSP_99.yml|tee logs/${file}.txt

# file=ATIS_tensor_2layers_FP32
# CUDA_VISIBLE_DEVICES=2 python -u tensor_train.py -tensorized 1 -uncompressed 0 -precondition 0 -save_model model/${file} -save_mode best -proj_share_weight -label_smoothing -batch_size 32 -epoch 500|tee logs/${file}.txt

# python -u tensor_train.py -tensorized 1 -uncompressed 0 -precondition 0 -save_model model/ATIS_tensor_2layers_FP32 -save_mode best -proj_share_weight -label_smoothing -batch_size 32 -epoch 500 |tee logs/ATIS_tensor_2layers_FP32.txt