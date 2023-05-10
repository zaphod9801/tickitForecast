# tickitForecast
This is a project that intend to be a test of how some predictive models like Sarimax works and its integration with FastAPI

## QuickStart Guide
First you have to clone the repository on your local machine and go into the root directory called "tickitForecast"
One you have done this it is recomendable to install virtual env in order to avoid possible dependencies and packages conflicts you can check it out this link explaining 
the installation and setup of vitualenv: https://virtualenv.pypa.io/en/latest/installation.html

Once you have your virtualenv running, you have to run the following commands in order to install dependencies and the project:
```bash
pip install -r requirements.txt
pip install -e .
```
It is important to run this commands in the root directory of the project. 
Now you can run the app locally using 

```bash
python app/main_start.py
```
This command will start a uvicorn server running in [localhost:8080]. If you want to check the endpoint you should browse to [localhost:8080/predict], this will call the two predictive
models inside the app and will give you the sales of the following 7 days (using the data provided in the url [https://docs.aws.amazon.com/redshift/latest/gsg/samples/tickitdb.zip])
The response should look like this:
![image](https://github.com/zaphod9801/tickitForecast/assets/71454879/2046d2b2-fdc0-4efc-8d7a-f26edb00b3be)

This endpoint just accept a GET request so keep it in mind.

Alternativately you also can run this with Docker, for this you first have to have Docker install in your system, check it out this guide about how to do it: https://www.docker.com/get-started/

Once you have Docker installed on your machine you just have to run the following two commands:
```bash
docker build -t [DESIRED_IMAGE_NAME]:latest .
```
This will build a Docker Image following the specifications in the dockerfile, so it is important to run this also in the root directory of the project.
Once the build finish, you run the next command:
```bash
docker run -d -p 8080:8080 [DESIRED_IMAGE_NAME]:latest
```
Now you can browse to [localhost:8080/predict] and check the endpoint working.

This project is currently temporaly deployed in the following URL: https://fastapi-tickit-app-tzfffgrx2a-ue.a.run.app/predict/ so you can check here the endpoint without doing anything 
of the above exposed, but only for the next week.

