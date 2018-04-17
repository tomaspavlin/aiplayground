
for a in 0.05 0 0.3 0.5 0.6 0.8 0.9; do
    echo Dropout $a
    python3 mnist_dropout.py --dropout $a
done
