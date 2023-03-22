def kfold(dataset, k=5):
    dataset_fold = []
    split_chunk = len(dataset) // 5

    for i in range(k):
        data = [dataset[i] for i in range(split_chunk * i, split_chunk * (i+1))]
        dataset_fold.append(data)
    
    for k_num in range(k):
        train_dataset = []
        for i in range(k):
            if i != k_num:
                train_dataset += dataset_fold[i]
        val_dataset = dataset_fold[k_num]
        yield train_dataset, val_dataset, split_chunk