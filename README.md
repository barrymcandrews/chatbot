# Chatbot
A Seq2seq chatbot written in python using tensorflow and keras.

### About
As the creator of a [chat website](https://github.com/barrymcandrews/raven-iac) and as an avid Westworld fan, I figured I'd take a try at creating my own chatbot.

The goal of this project was to train a chatbot using my text messages, so that the chatbot would resemble the way I speak. Along the way, I discovered that my text messages were not an amazing dataset, so I pulled in the [Cornell Movie Dialogs](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) which seems to be very popular for chatbots.

## Setup
To setup the project, clone this repository and run the `setup.py` file:

```
$ python3 setup.py develop
```

## Training
You can use the `train` command to train the chatbot model. This command will assemble everything your computer needs to begin training.

The `train` command will do the following:

1. Download the [Cornell Movie Dialogs](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset.
2. Download the [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/) for vector representations of words.
3. Tokenize the dataset
4. Train the model

The initial downloads can take some time to complete, but, you'll only need to download them the first time you run the command.

```
$ bot train --epochs 100
```
#### Available Options:
* `--epochs <int>` (Default: 10)
* `--learning-rate <int>` (Default: 0.01)
* `--batch-size <int>` (Default: 128)
* `--build-dir <path>` (Default: "./build")
* `--k-folds` (Default: 10)
* `--upload`



## Chatting
Once you have trained the chatbot model, you can use the `chat` command to test the chatbot. This command starts a basic interaction loop. You can type in a response and see how the chatbot will respond.

```
$ bot chat
```
#### Available Options:
* `--build-dir <path>` (Default: "./build")


## Build Management
In order to help with organizing models trained with different types of data, I created a build management system for this project. All the builds are stored in an S3 bucket, and can be managed using the command line.
Here are the highlights:

* `bot builds list` — list all models stored in the bucket
* `bot builds put` — upload the model that was just trained
* `bot builds get` — download a previously trained model
* `bot builds delete` — delete a model

You can also automatically upload a build after training by including the `--upload` flag. This can be useful when submitting a headless job on a more powerful computer such as GCP's [AI Platform](https://cloud.google.com/ai-platform).

```
$ bot train --upload
```

#### Cloud Configuration
For now the S3 bucket is hardcoded into `builds.py` you'll have to configure it there. You'll also need to include your AWS credentials in either `builds.py` or your local `~/.aws/configure`
