# Movie Classifier

### Overview

A baseline Movie Classifier allows the user to predict the genre of a given film, based on a title of the movie and its description. 

- Language: Python 3.6.1
- ML Libraries: Scikit-learn, scikit-multilearn

The programming language and ML libraries:
- Are very flexible Allow for easy implementation;
- Support methodologies suited for multi-label classification;
- In general suit the rapid prototyping nature of the task.

### Data
The dateset can be downloaded from [here](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv) and stored in `src/resources/data/the-movies-dataset`.


### Prepare environement
- [Install pip](https://www.shellhacks.com/python-install-pip-mac-ubuntu-centos/);
- Install virtualenv: `pip3 install virtualenv`;
- Create new environment: `virtualenv -p /usr/bin/python3.6 mc_env`;
- Switch to now environment: `source mc_env/bin/activate`;
- Install dependences: `pip3 install --requirement requirements.txt`.
### Main API
By default, there are pre-trained class label map, tfidf vectorizer and a Random Forest model in `src/resouces/models`.
If the user wants to sue the default configuration, run:
`python movie_classifier.py ---title <title> --description <description>`
This predicts the movie genre and outputs it to console and stories it in `src/resouces/models`.
Alternatively, if some of the pre-trained objects are missing or if the following command is executed `python movie_classifier.py ---title <title> --description <description> --retrain`, a new model is trained, given the default parameters. For reproducibility, random state is set to `42` in the classifier and the cross validation.

### Docker
To run the model inside a Docker container:
- Install Docker: `https://docs.docker.com`;
- Run: `docker build . -t movie_classifier` to build container;
- Run: `docker run -it -d --name <container-name> <image-name> bash` to run container;
- Run: `docker exec -it <container-name> bash` to execute into the container;
- Once inside the container, follow main API.