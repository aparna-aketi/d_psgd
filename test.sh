
# python trainer.py  --data-dir ../../data/imagenette   --lr=0.1  --batch-size=160  --world_size=5 --skew=1 --gamma=0.5 --normtype=evonorm --optimizer=ngc --epoch=100 --arch=resnet --depth=18 --momentum=0.9 --alpha 1.0 --dataset=imagenette_full --graph=chain --seed=32

############################

# python trainer.py --lr=0.1 --momentum=0.9 --batch-size=512 --world_size=16 --graph=ring --neighbors=2 --arch=resnet --alpha=1 --epochs=100 --gamma=1.0 --seed=1234 
# cd ./outputs
# python dict_to_csv.py --arch=resnet --world_size=16 --graph=ring --gamma=1.0 --alpha=1.0 --seed=1234
# cd ..

python trainer.py --lr=0.1 --momentum=0.9 --batch-size=512 --world_size=16 --graph=ring --neighbors=2 --arch=resnet --alpha=1 --epochs=100 --gamma=0.5 --seed=1234 
cd ./outputs
python dict_to_csv.py --arch=resnet --world_size=16 --graph=ring --gamma=0.5 --alpha=1.0 --seed=1234
cd ..

python trainer.py --lr=0.1 --momentum=0.9 --batch-size=512 --world_size=16 --graph=ring --neighbors=2 --arch=resnet --alpha=1 --epochs=100 --gamma=0.1 --seed=1234 
cd ./outputs
python dict_to_csv.py --arch=resnet --world_size=16 --graph=ring --gamma=0.1 --alpha=1.0 --seed=1234
cd ..
###############################

python trainer.py --lr=0.1 --momentum=0.9 --batch-size=512 --world_size=16 --graph=ring --neighbors=2 --arch=resnet --alpha=0.1 --epochs=100 --gamma=1.0 --seed=1234 
cd ./outputs
python dict_to_csv.py --arch=resnet --world_size=16 --graph=ring --gamma=1.0 --alpha=0.1 --seed=1234
cd ..

python trainer.py --lr=0.1 --momentum=0.9 --batch-size=512 --world_size=16 --graph=ring --neighbors=2 --arch=resnet --alpha=0.1 --epochs=100 --gamma=0.5 --seed=1234 
cd ./outputs
python dict_to_csv.py --arch=resnet --world_size=16 --graph=ring --gamma=0.5 --alpha=0.1 --seed=1234
cd ..

python trainer.py --lr=0.1 --momentum=0.9 --batch-size=512 --world_size=16 --graph=ring --neighbors=2 --arch=resnet --alpha=0.1 --epochs=100 --gamma=0.1 --seed=1234 
cd ./outputs
python dict_to_csv.py --arch=resnet --world_size=16 --graph=ring --gamma=0.1 --alpha=0.1 --seed=1234
cd ..
###############################