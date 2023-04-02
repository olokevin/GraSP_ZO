# CUDA_VISIBLE_DEVICES=1 python -u tensor_train.py -tensorized 1 -uncompressed 0 -precondition 0 -n_warmup_steps 50000 -data data/multi30k.atok.low.pt -save_model model/test_ATIS_tensor_2layer_INT4 -save_mode best -proj_share_weight -label_smoothing -batch_size 64 -epoch 500|tee logs/test_ATIS_tensor_2layer_INT4.txt

# ================== TTM, FO ==================
# file=ATIS_TTM_FO_95
# configs=configs/ATIS/TTM/FO.yml
# CUDA_VISIBLE_DEVICES=2 python -u tensor_train.py \
# -tensorized 1 -uncompressed 0 -precondition 0 -save_mode best \
# -proj_share_weight -label_smoothing -batch_size 64 -config ${configs} \
# |tee logs/ATIS/TTM/summary/${file}.txt

# ================== TTM, SCD incre ==================
# file=ATIS_TTM_SCD_95_incre
# configs=configs/ATIS/TTM/SCD.yml
# CUDA_VISIBLE_DEVICES=2 python -u tensor_train.py \
# -tensorized 1 -uncompressed 0 -precondition 0 -save_mode best \
# -proj_share_weight -label_smoothing -batch_size 64 -config ${configs} \
# |tee logs/ATIS/TTM/summary/${file}.txt

# ================== GraSP, FO ==================
# file=ATIS_GraSP_FO_95
# configs=configs/ATIS/GraSP/FO.yml
# CUDA_VISIBLE_DEVICES=2 python -u tensor_train.py \
# -tensorized 0 -uncompressed 0 -precondition 0 -save_mode best \
# -proj_share_weight -label_smoothing -batch_size 64 -config ${configs} \
# |tee logs/ATIS/GraSP/summary/${file}.txt

# ================== GraSP, SCD sign ==================
# file=scratch_95_grad0.5_lr0.001
configs=configs/ATIS/GraSP/SCD.yml
CUDA_VISIBLE_DEVICES=2 python -u tensor_train.py\
 -tensorized 0 -uncompressed 0 -precondition 0 -save_mode best\
 -proj_share_weight -label_smoothing -batch_size 64 -config ${configs}
#  |tee logs/ATIS/GraSP/summary/SCD_sign/${file}.txt

# ================== GraSP, SCD batch ==================
# # file=scratch_95_grad0.8_lr0.001
# configs=configs/ATIS/GraSP/SCD_batch.yml
# CUDA_VISIBLE_DEVICES=2 python -u tensor_train.py\
#  -tensorized 0 -uncompressed 0 -precondition 0 -save_mode best\
#  -proj_share_weight -label_smoothing -batch_size 64 -config ${configs}
# #  |tee logs/ATIS/GraSP/summary/SCD_batch/${file}.txt

# ================== GraSP, SGD ==================
# file=ATIS_GraSP_SGD_95_gamma09_incre
# configs=configs/ATIS/GraSP/SGD.yml
# CUDA_VISIBLE_DEVICES=2 python -u tensor_train.py \
# -tensorized 0 -uncompressed 0 -precondition 0 -save_mode best \
# -proj_share_weight -label_smoothing -batch_size 64 -config ${configs} \
# |tee logs/ATIS/GraSP/summary/${file}.txt

# file=ATIS_tensor_2layers_FP32
# CUDA_VISIBLE_DEVICES=2 python -u tensor_train.py -tensorized 1 -uncompressed 0 -precondition 0 -save_model model/${file} -save_mode best -proj_share_weight -label_smoothing -batch_size 32 -epoch 500|tee logs/${file}.txt
