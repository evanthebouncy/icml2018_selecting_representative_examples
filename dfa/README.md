# To run:
The model comes with a pre-trained model /models/m1

To run the model and compare it against other baselines:

    python time_experiment.py

It will periodically generate a data file \_result.p

# To plot:

Rename the result file:

    cp \_result.p result.p
    python grapher.py

# To train the model

If you want to train your own model, run:

    python model.py

By default it will be saved under /models/tmp periodically

To use this model move it to /models/m1

