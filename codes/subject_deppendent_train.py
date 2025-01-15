from torcheeg.model_selection import KFoldPerSubjectGroupbyTrial,KFoldPerSubjectCrossTrial
import os 
from  pathlib import Path
from torch.utils.data import DataLoader
from copy import copy,deepcopy

from torcheeg.trainers import ClassifierTrainer
from dwc_codes.DualClassifierTrainer import DLClassifierTrainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
from dwc_codes.SEALtrainer import SEALClassifierTrainer

def train_seal_per_subject(dataset,model, 
                           split_path ='./tmp_out/split_path',
                           seal_epoch=1,
                           metrics=['accuracy','kappa','f1score']):
    if dataset.__class__.__name__ == 'StrokePatientsMIDataset':
        KF = KFoldPerSubjectGroupbyTrial(shuffle=True,split_path='./tmp_out/split_path' ,n_splits=4)
    else:
        KF = KFoldPerSubjectGroupbyTrial(shuffle=True,split_path='./tmp_out/split_path_hos',n_splits=4,random_state=42)
    _model = model
    subjects = dataset.info['subject_id'].unique()
    print(subjects)
    subjects.sort()
    for subject in subjects:
        directory = Path(f'subject_dependent/{dataset.__class__.__name__}/SEAL_{model.__class__.__name__}_checkpoints/S{subject}')
        already_saved = []
        if os.path.exists(str((directory.absolute()))):
            import re
            pattern = "(\d)fold.*.ckpt"
            already_saved = [ re.findall(pattern,file.name)[0] for file in directory.iterdir() if re.findall(pattern,file.name)]
        print('already_save_fold',already_saved)

        for i,(train_dataset,test_dataset) in enumerate(KF.split(dataset=dataset,subject=f'{subject}')):
            if str(i) in already_saved:
                print(f'subject{subject} -{i} fold had been saved, pass.')
                continue
            print(f'fitting subjec{subject}_fold{i}!')
            train_data,test_data = DataLoader(train_dataset,batch_size=32,shuffle=True),DataLoader(test_dataset,batch_size=32)
            
            ckpt_name = f'{i}fold_'+'best-{epoch:02d}' 
            for metric_name in metrics:
                ckpt_name += "-{val_"+f"{metric_name}"+":.3f}"
           
            model = deepcopy(_model)
            save_callbacks = ModelCheckpoint(
                monitor='val_accuracy',     
                mode='max',             
                save_top_k=1,           
                save_last=True,         
                every_n_epochs=1,      
                dirpath=str(directory.absolute()), 
                filename=ckpt_name
            )
            # 设置EarlyStopping回调
            early_stopping = EarlyStopping(
                monitor='val_accuracy',       
                patience=20,              
                verbose=True,             
                mode='max'       
            )
            ct = SEALClassifierTrainer(model= model,num_classes=2, accelerator='gpu',devices=4,metrics=metrics)
            train_data = ct.prepare_train_dataloader(train_dataset,batch_size=32,shuffle=True)
            ct.fit(train_data,test_data,callbacks= [save_callbacks,early_stopping],seal_update_epochs=seal_epoch)

def kfold_train_seal(dataset,model,seal_epoch=1,metrics=['accuracy','kappa','f1score']):
    from torcheeg.model_selection import KFold
    
    from torch.utils.data import DataLoader
    from torcheeg.trainers import ClassifierTrainer
    
    from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

    directory = Path(f'5fold/{dataset.__class__.__name__}/SEAL-{model.__class__.__name__}')
    already_saved = []
    if os.path.exists(str((directory.absolute()))):
        import re
        pattern = "(\d)fold.*.ckpt"
        already_saved = [ re.findall(pattern,file.name)[0] for file in directory.iterdir() if re.findall(pattern,file.name)]
    print('already_save_fold',already_saved)


    kfold = KFold(shuffle=True)
    for i, (train_dataset,test_dataset) in enumerate(kfold.split(dataset=dataset)):
        train_data = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
        test_data = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
        
        model_ = deepcopy(model)

        save_callbacks = ModelCheckpoint(
            monitor='val_accuracy',     
            mode='max',             
            save_top_k=1,           
            save_last=True,         
            every_n_epochs=1,      
            dirpath=str(directory.absolute()), 
            filename=f'{i}fold_'+'best-{epoch:02d}-{val_accuracy:.2f}' 
        )
        # 设置EarlyStopping回调
        early_stopping = EarlyStopping(
            monitor='val_accuracy',       
            patience=35,              
            verbose=True,             
            mode='max'       
        )
        ct = SEALClassifierTrainer(model=model_,num_classes=2,devices=4,accelerator='gpu')
        train_data = ct.prepare_train_dataloader(train_dataset,batch_size=32,shuffle=True)
        ct.fit(train_data,test_data,callbacks =[save_callbacks,early_stopping],seal_update_epochs=seal_epoch)



def train(dataset,model,split_path ='./tmp_out/split_path',collate_fn=None,cross_trial=False):
    if not cross_trial:
        if dataset.__class__.__name__ == 'StrokePatientsMIDataset' and split_path is not None:
            KF = KFoldPerSubjectGroupbyTrial(shuffle=True,split_path=split_path ,n_splits=4)
        else:
            KF = KFoldPerSubjectGroupbyTrial(shuffle=True,split_path='./tmp_out/split_path_hos',n_splits=5,random_state=42)
    else:
        if dataset.__class__.__name__ == 'StrokePatientsMIDataset' and split_path is not None:
            KF = KFoldPerSubjectCrossTrial(shuffle=True,split_path=split_path ,n_splits=4)
        else:
            KF = KFoldPerSubjectCrossTrial(shuffle=True,split_path='./tmp_out/split_path_hos',n_splits=5,random_state=42)
    _model = model
    subjects = dataset.info['subject_id'].unique()
    subjects.sort()
    for subject in subjects:
        directory = Path(f'subject_dependent/{dataset.__class__.__name__}/{model.__class__.__name__}_checkpoints/S{subject}')
        already_saved = []
        if os.path.exists(str((directory.absolute()))):
            import re
            pattern = "(\d)fold.*.ckpt"
            already_saved = [ re.findall(pattern,file.name)[0] for file in directory.iterdir() if re.findall(pattern,file.name)]
        print('already_save_fold',already_saved)

        for i,(train_dataset,test_dataset) in enumerate(KF.split(dataset=dataset,subject=f'{subject}')):
            if str(i) in already_saved:
                print(f'subject{subject} -{i} fold had been saved, pass.')
                continue
            print(f'fitting subjec{subject}_fold{i}!')
            train_data,test_data = DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=collate_fn),\
                DataLoader(test_dataset,batch_size=32,collate_fn=collate_fn)
            
            model = deepcopy(_model)

            save_callbacks = ModelCheckpoint(
                monitor='val_accuracy',     
                mode='max',             
                save_top_k=1,           
                save_last=True,         
                every_n_epochs=1,      
                dirpath=str(directory.absolute()), 
                filename=f'{i}fold_'+'best-{epoch:02d}-{val_accuracy:.2f}' 
            )
            # 设置EarlyStopping回调
            early_stopping = EarlyStopping(
                monitor='val_accuracy',       
                patience=35,              
                verbose=True,             
                mode='max'       
            )
            
            ct = ClassifierTrainer(model= model,num_classes=2, accelerator='gpu',devices=4)
            ct.fit(train_data,test_data,callbacks= [save_callbacks,early_stopping])




def train_sd_flip(dataset,dataset_flip,model,split_path ='./tmp_out/split_path' ):
    from torch.utils.data import Dataset, DataLoader

    class MixedDataset(Dataset):
        def __init__(self, dataset1, dataset2):
            self.dataset1 = dataset1
            self.dataset2 = dataset2
            self.length1 = len(dataset1)
            self.length2 = len(dataset2)

        def __len__(self):
            return self.length1 + self.length2

        def __getitem__(self, idx):
            if idx < self.length1:
                return self.dataset1[idx]
            else:
                return self.dataset2[idx - self.length1]
          


    if dataset.__class__.__name__ == 'StrokePatientsMIDataset':
        KF = KFoldPerSubjectGroupbyTrial(shuffle=True,split_path='./tmp_out/split_path' ,n_splits=4)
    else:
        KF = KFoldPerSubjectGroupbyTrial(shuffle=True,split_path='./tmp_out/split_path_hos',n_splits=5,random_state=42)
    _model = model
    subjects = dataset.info['subject_id'].unique()
    subjects.sort()
    for subject in subjects:
        directory = Path(f'subject_dependent/{dataset.__class__.__name__}/Hflip_{model.__class__.__name__}_checkpoints/S{subject}')
        already_saved = []
        if os.path.exists(str((directory.absolute()))):
            import re
            pattern = "(\d)fold.*.ckpt"
            already_saved = [ re.findall(pattern,file.name)[0] for file in directory.iterdir() if re.findall(pattern,file.name)]
        print('already_save_fold',already_saved)

        for i,((train_dataset,test_dataset),(train_dataset2,test_dataset2)) in enumerate(zip(KF.split(dataset=dataset,subject=f'{subject}'),KF.split(dataset=dataset_flip,subject=f'{subject}'))):
            if str(i) in already_saved:
                print(f'subject{subject} -{i} fold had been saved, pass.')
                continue
            print(f'fitting subjec{subject}_fold{i}!')

            mix_train_dataset = MixedDataset(train_dataset,train_dataset2)
            train_data,test_data = DataLoader(mix_train_dataset,batch_size=32,shuffle=True),DataLoader(test_dataset,batch_size=32)

            model = copy(_model)
             
            save_callbacks = ModelCheckpoint(
                monitor='val_accuracy',     
                mode='max',             
                save_top_k=1,           
                save_last=True,         
                every_n_epochs=1,      
                dirpath=str(directory.absolute()), 
                filename=f'{i}fold_'+'best-{epoch:02d}-{val_accuracy:.2f}' 
            )
            # 设置EarlyStopping回调
            early_stopping = EarlyStopping(
                monitor='val_accuracy',       
                patience=35,              
                verbose=True,             
                mode='max'       
            )
            ct = ClassifierTrainer(model= model,num_classes=2, accelerator='gpu',devices=4)
            ct.fit(train_data,test_data,callbacks= [save_callbacks,early_stopping])
           


def train_groupby_para(dataset,model,split_path ='./tmp_out/split_path' ):
    if dataset.__class__.__name__ == 'StrokePatientsMIDataset':
        KF = KFoldPerSubjectGroupbyTrial(shuffle=True,split_path='./tmp_out/split_path' ,n_splits=4)
    else:
        KF = KFoldPerSubjectGroupbyTrial(shuffle=True,split_path='./tmp_out/split_path_hos',n_splits=5,random_state=42)
    _model = model
    subjects = dataset.info['subject_id'].unique()
    subjects.sort()
    for subject in subjects:
        directory = Path(f'subject_dependent/{dataset.__class__.__name__}/Dual_{model.__class__.__name__}_checkpoints/S{subject}')
        already_saved = []
        if os.path.exists(str((directory.absolute()))):
            import re
            pattern = "(\d)fold.*.ckpt"
            already_saved = [ re.findall(pattern,file.name)[0] for file in directory.iterdir() if re.findall(pattern,file.name)]
        print('already_save_fold',already_saved)

        for i,(train_dataset,test_dataset) in enumerate(KF.split(dataset=dataset,subject=f'{subject}')):
            if str(i) in already_saved:
                print(f'subject{subject} -{i} fold had been saved, pass.')
                continue
            print(f'fitting subjec{subject}_fold{i}!')
            train_data,test_data = DataLoader(train_dataset,batch_size=32,shuffle=True),DataLoader(test_dataset,batch_size=32)
            
            model = copy(_model)
             

            save_callbacks = ModelCheckpoint(
                monitor='val_accuracy',     
                mode='max',             
                save_top_k=1,           
                save_last=True,         
                every_n_epochs=1,      
                dirpath=str(directory.absolute()), 
                filename=f'{i}fold_'+'best-{epoch:02d}-{val_accuracy:.2f}' 
            )
            # 设置EarlyStopping回调
            early_stopping = EarlyStopping(
                monitor='val_accuracy',       
                patience=35,              
                verbose=True,             
                mode='max'       
            )
            ct = DLClassifierTrainer(model= model,num_classes=2, accelerator='gpu',devices=4)
            ct.fit(train_data,test_data,callbacks= [save_callbacks,early_stopping])



def kfold_train(dataset,model):
    from torcheeg.model_selection import KFold
    
    from torch.utils.data import DataLoader
    from torcheeg.trainers import ClassifierTrainer
    from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

    directory = Path(f'5fold/{dataset.__class__.__name__}/result-{model.__class__.__name__}')
    already_saved = []
    if os.path.exists(str((directory.absolute()))):
        import re
        pattern = "(\d)fold.*.ckpt"
        already_saved = [ re.findall(pattern,file.name)[0] for file in directory.iterdir() if re.findall(pattern,file.name)]
    print('already_save_fold',already_saved)



    kfold = KFold(shuffle=True)
    for i, (train_dataset,test_dataset) in enumerate(kfold.split(dataset=dataset)):
        train_data = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
        test_data = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
        
        model_ = deepcopy(model)

        save_callbacks = ModelCheckpoint(
            monitor='val_accuracy',     
            mode='max',             
            save_top_k=1,           
            save_last=True,         
            every_n_epochs=1,      
            dirpath=str(directory.absolute()), 
            filename=f'{i}fold_'+'best-{epoch:02d}-{val_accuracy:.2f}' 
        )
        # 设置EarlyStopping回调
        early_stopping = EarlyStopping(
            monitor='val_accuracy',       
            patience=35,              
            verbose=True,             
            mode='max'       
        )
        ct = ClassifierTrainer(model=model_,num_classes=2,devices=4,accelerator='gpu',weight_decay=1e-4)
        ct.fit(train_data,test_data,callbacks =[save_callbacks,early_stopping] )



def loso_train(model,dataset):
    from torcheeg.model_selection import LeaveOneSubjectOut
    from torch.utils.data import DataLoader
    from torcheeg.trainers import ClassifierTrainer
    from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping

    _model = deepcopy(model)

    loso = LeaveOneSubjectOut()
    for i, (train_dataset,test_dataset) in enumerate(loso.split(dataset=dataset)):
        subject = test_dataset.info['subject_id'].values[-1]
        directory = Path(f'loso/{dataset.__class__.__name__}/{model.__class__.__name__}/{subject}')
        already_saved = []
        if os.path.exists(str((directory.absolute()))):
            import re
            pattern = "(\d)fold.*.ckpt"
            already_saved = [ re.findall(pattern,file.name)[0] for file in directory.iterdir() if re.findall(pattern,file.name)]
        print('already_save_fold',already_saved)

        model = deepcopy(_model)

        train_data = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
        test_data = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
        
        save_callbacks = ModelCheckpoint(
            monitor='val_accuracy',     
            mode='max',             
            save_top_k=1,           
            save_last=True,         
            every_n_epochs=1,      
            dirpath=str(directory.absolute()), 
            filename=f'{i}fold_'+'best-{epoch:02d}-{val_accuracy:.2f}' 
        )
        # 设置EarlyStopping回调
        early_stopping = EarlyStopping(
            monitor='val_accuracy',       
            patience=35,              
            verbose=True,             
            mode='max'       
        )
        ct = ClassifierTrainer(model=model,num_classes=2,devices=4,accelerator='gpu')
        ct.fit(train_data,test_data,callbacks =[save_callbacks,early_stopping] )