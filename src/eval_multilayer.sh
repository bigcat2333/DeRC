for i in 0 1 2 3 4 5 6 7 8 9 10 11
do
	echo $i
	str="'s/logits = logits_list\[-1\]/logits = logits_list\[$i\]/g'"
	cmd="sed -ie $str bert_distill.py"
	eval $cmd
	CUDA_VISIBLE_DEVICES=0 python train_distill_bert.py --output_dir /path/to/save/checkpoints/baseline_mnli_12classifiers/model_55999 --do_eval --mode none --seed 111 --which_bias hans --dataset mnli --num_train_epochs 5
	str="'s/logits = logits_list\[$i\]/logits = logits_list\[-1\]/g'"
	cmd="sed -ie $str bert_distill.py"
	eval $cmd
done
