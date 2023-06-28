
for i in model_11592
do
	echo $i
	CUDA_VISIBLE_DEVICES=2 python train_distill_bert.py    --output_dir /path/to/save/checkpoints/fever/$i --do_eval --mode none   --seed 111 --which_bias hans --dataset fever
done
