for i in model_45999 model_46499 model_46999 model_47499 model_47999 model_48499 model_48999 model_49499 model_49999 model_50499 model_50999 model_51499 model_51999 model_52499 model_52999 model_53499 model_53999 model_54499 model_54999 model_55499 model_55999 model_56499
do
	echo $i
	CUDA_VISIBLE_DEVICES=2 python train_distill_bert.py --output_dir /path/to/save/checkpoints/1228_DeRC_layer3_QQP/$i --do_eval --mode none --seed 111 --which_bias hans --dataset QQP --num_train_epochs 5
done
