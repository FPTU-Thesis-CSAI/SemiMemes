for dataset in train_images val_images test_images
do	
	python feature_extraction/lxmert/extract_img_features.py \
		--img_folder data/Memotion2.0/images/$dataset \
		--output_folder_path data/features/lxmert/$dataset
done
