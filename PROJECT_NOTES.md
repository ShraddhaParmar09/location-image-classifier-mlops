# 📍 Location Image Classifier — MLOps

> **Purpose:** Personal learning notes and technical understanding for my MLOps portfolio project.

---

# -[x] Phase 0 — Understanding the Project


**1. A. What is the problem?**

-> project named Location image classifire.
-> in this project what we are doing is the user have so many places images and want to know the place.
-> so what user doing is upload the image and our model will predict the image location and then gives a location name. 

> example: User uploads an image → the ML model analyzes visual features → predicts one of the known location classes → returns the predicted location.

**2. Why is machine learning appropriate?**

-> becoz right now we are using technology.. like phone, tablet, pc, laptop and everything we are working on machine. 
-> so, if i am asking questions to machine so, machine also have to understand the question and then they will give the answer.
-> machine learning is all about having a large amount of data so, using that data we have to train machine to give answer accroading to the question right.
-> so, here if we talk about my project, my machine have to take large amount of different location image data and accroading to that data we have to train them then machine will give result.
-> so that why machine learning is important part.

> Machine learning:
Instead of manually writing rules for every possible image, we provide examples and let the model learn patterns from those examples.

> For our project:

There can be many visual characteristics associated with different locations—buildings, architecture, landscapes, signs, colors, structures, etc. It would be difficult to manually write rules for every possible image. A machine-learning model can learn patterns from a labeled dataset of location images and use those learned patterns to predict the location of a new image.

**3. What is MLOps?**

-> MLops -> machine learning + Devops
-> both the working together and focusing on process of deploying machine learning models for production and then maintaing and monitoring them.

> It covers the lifecycle around an ML system:
development → training → testing → deployment → monitoring → maintenance → retraining/versioning

**4. Why would a trained ML model need an MLOps pipeline?**

-> automation
-> scalability
-> reproducibility
-> monitoring
-> governance

> Imagine:

You trained a model today.

Then tomorrow:

code changes
model changes
dataset changes
dependency changes
model performance drops
you need to deploy a new version
10 users become 10,000 users

Doing everything manually becomes unreliable.

That's where MLOps comes in.

**5. What does Docker solve?**

-> its allows user to build, run and manage container.
-> and here container means its a type of software that packages up an application and all its dependencies so the application can run reliably from one computing environment to another.

**6. What does Kubernetes solve?**

->Imagine a scenario where you have to run multiple Docker containers, on multiple machines, to support an enterprise-level ML application with varied workloads day and night. As simple as it may sound, it is a lot of work to do manually.
->You need to start the right containers at the right time, figure out how they can talk to each other, handle storage considerations, and deal with failed containers or hardware. This is the problem Kubernetes is solving, by allowing large numbers of containers to work together in harmony, and reducing the operational burden.

> Docker = containerization
> Kubernetes = container orchestration

**7. What does CI/CD solve?**

-> continous integration 
-> continous delivery
-> continous deployment

**8. What role does Google Cloud play?**

-> The exact GCP services/configuration should be determined as we build rather than memorized upfront.

**9. What role does GitHub play?**

-> here, in github we publish our whole repositary. 
-> which include everything like our project datasets, models, learning notes, application and everything.

> GitHub's major role here is:

source-code hosting + version control + collaboration + project history

And in our project, GitHub will also act as the trigger/source for our CI/CD pipeline.

For example:

You change code
      ↓
git commit
      ↓
git push
      ↓
GitHub
      ↓
CI/CD pipeline starts

**10. What happens from git push to a user receiving a prediction?**

-> From the exact moment a Machine Learning Engineer types git push to when an end-user receives a live prediction via an API call, the code and models undergo a highly automated MLOps lifecycle.
-> This transition bridges two distinct software architectures: the Continuous Integration/Continuous Deployment (CI/CD) Pipeline and the Production Inference Infrastructure.

> Imagine you change something in your project.

You do:

git add .
git commit
git push

Now your code reaches GitHub.

Without CI/CD, you might have to manually do:

GitHub
   ↓
Download latest code
   ↓
Install dependencies
   ↓
Run tests
   ↓
Build application
   ↓
Build Docker image
   ↓
Push Docker image
   ↓
Deploy application

That's annoying and, more importantly, easy to make mistakes.

With CI/CD, we want the system to do these steps automatically.

So:

              YOU
               │
           git push
               ↓
            GitHub
               │
               ↓
        CI/CD Pipeline
               │
       ┌───────┴────────┐
       ↓                ↓
    Build             Test
       │                │
       └───────┬────────┘
               ↓
        Build Docker image
               ↓
        Push image to registry
               ↓
          Deploy/update
               ↓
       Running application
So what does CI actually do?

CI = Continuous Integration

When you push your changes, CI can automatically:

get your latest code
install dependencies
run tests
check whether the application builds successfully
catch problems early

Basically:

“Did my new code break the project?”

# -[ ] Phase 1 — Research & Requirements 🔎

> 1. Problem research ->> docs/research.md

> 2. Existing solutions ->> docs/research.md

> 3. Dataset research ->> docs/dataset-analysis.md

> 4. Define requirements ->> docs/requirements.md

> 5. Define project scope ->> docs/requirements.md

> 6. Technical decision ->> docs/technical-plan.md





