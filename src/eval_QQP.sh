
for i in model_1000   model_13371  model_17371  model_20371  model_24742  model_28742  model_31742  model_36113  model_4000   model_43113  model_47484  model_50484  model_54484  model_7000  model_10000  model_14371  model_18371  model_21371  model_25742  model_29742  model_32742  model_37113  model_40113  model_44113  model_48484  model_51484  model_55484  model_8000  model_11000  model_15371  model_19371  model_22371  model_26742  model_3000   model_33742  model_38113  model_41113  model_45113  model_49484  model_52484  model_56484  model_9000  model_12371  model_16371  model_2000   model_23742  model_27742  model_30742  model_35113  model_39113  model_42113  model_46484  model_5000   model_53484  model_6000
do
	echo $i
	CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py   --output_dir /path/to/save/checkpoints/paws_residual_connection_5epochs/$i --do_eval --mode none   --seed 111 --which_bias hans --dataset QQP
done
