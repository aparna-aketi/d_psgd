############################

python trainer.py --lr=0.1 --momentum=0.9 --batch-size=512 --world_size=16 --graph=ring --neighbors=2 --arch=resnet --alpha=1 --epochs=100 --gamma=1.0 --seed=1234 --dataset=cifar10 --classes=10
cd ./outputs
python dict_to_csv.py --arch=resnet --world_size=16 --graph=ring --gamma=1.0 --alpha=1.0 --seed=1234
cd ..

###############################
