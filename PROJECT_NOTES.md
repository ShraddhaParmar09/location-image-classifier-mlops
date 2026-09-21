# 📍 Location Image Classifier — MLOps

> **Purpose:** Personal learning notes and technical understanding for my MLOps portfolio project.

---

# ☑️ Phase 0 — Understanding the Project

## **1. What is the problem?**

→ Project named **Location Image Classifier**.

→ In this project, a user may have an image of a place and want to know the location shown in that image.

→ The user uploads the image, and our ML model analyzes it and predicts the location.

> **Example:**
>
> User uploads an image → the ML model analyzes visual features → predicts one of the known location classes → returns the predicted location.

### **Important understanding**

This project is **not trying to identify every possible location on Earth** in V1.

The V1 system will work with a **predefined set of location classes**. The model learns from images belonging to those classes and predicts which known class a new image belongs to.

**Example:**

```
```

```
Input image
     ↓
Image preprocessing
     ↓
Trained ML model
     ↓
Predicted location class
     ↓
Location + confidence
```

---

## **2. Why is machine learning appropriate?**

→ Because right now we are using technology like phones, tablets, PCs, laptops, etc., and we interact with machines to get results.

→ If we give a question or input to a machine, the machine needs some way to process that input and produce an answer.

→ Machine learning uses data to train a model so that the model can learn patterns from examples and make predictions on new data.

→ So, here, my machine has to take a large amount of different location image data and, according to that data, we train the model. Then the trained model can predict the location of a new image.

→ That's why machine learning is an important part of this project.

> **Machine learning:**
>
> Instead of manually writing rules for every possible image, we provide examples and let the model learn patterns from those examples.

> **For our project:**
>
> There can be many visual characteristics associated with different locations—buildings, architecture, landscapes, signs, colors, structures, etc. It would be difficult to manually write rules for every possible image. A machine-learning model can learn patterns from a labeled dataset of location images and use those learned patterns to predict the location of a new image.

### **Important distinction**

```
```

```
Training:
Images + known location labels
             ↓
          ML model
             ↓
       learns patterns

Inference:
New unseen image
       ↓
Trained ML model
       ↓
Predicted location
```

**Training and inference are different stages.**

---

## **3. What is MLOps?**

→ MLOps = **Machine Learning + Operations/DevOps practices**

→ It focuses on managing the lifecycle of machine-learning systems, including development, training, testing, deployment, monitoring, maintenance, and updating/versioning.

> **It covers the lifecycle around an ML system:**

```
```

```
development
     ↓
data preparation
     ↓
training
     ↓
testing/evaluation
     ↓
deployment
     ↓
monitoring
     ↓
maintenance
     ↓
retraining / versioning
```

### **Important understanding**

Building an ML model is only one part of an ML project.

For example:

```
```

```
Model works on my laptop
        ≠
Reliable ML application in production
```

MLOps helps connect the ML development process with reliable software and production operations.

---

## **4. Why would a trained ML model need an MLOps pipeline?**

Main reasons:

- **Automation** 
- **Scalability** 
- **Reproducibility** 
- **Monitoring** 
- **Versioning** 
- **Reliable deployment** 
- **Maintenance** 

> **Imagine:**

You trained a model today.

Then tomorrow:

-  code changes 
-  model changes 
-  dataset changes 
-  dependency changes 
-  model performance drops 
-  you need to deploy a new version 
-  10 users become 10,000 users 

Doing everything manually becomes unreliable.

That's where MLOps comes in.

### **Simple understanding**

Without MLOps:

```
```

```
Developer
   ↓
Manual training
   ↓
Manual testing
   ↓
Manual deployment
   ↓
Manual monitoring
```

With MLOps:

```
```

```
Code / Data / Model changes
          ↓
     Automated process
          ↓
 Testing → Build → Deploy
          ↓
       Monitoring
          ↓
     Update / Retrain
```

---

## **5. What does Docker solve?**

→ Docker allows us to build, run, and manage **containers**.

→ A container packages an application together with the dependencies it needs so that it can run consistently across different computing environments.

### **Why is this useful for our project?**

Our ML application may depend on:

-  Python version 
-  ML libraries 
-  API framework 
-  system libraries 
-  model files 
-  application code 

Without proper packaging, an application might work on one machine but fail on another because the environments are different.

Docker helps package the application environment consistently.

> **Simple understanding:**

```
```

```
Application
     +
Dependencies
     +
Configuration
     ↓
   Docker
     ↓
Container
```

> **Docker = containerization**

---

## **6. What does Kubernetes solve?**

Imagine a scenario where you have to run multiple Docker containers on multiple machines to support an enterprise-level ML application with different workloads throughout the day.

You may need to:

-  start the right containers 
-  manage multiple containers 
-  handle failed containers 
-  scale applications 
-  manage communication between services 
-  distribute workloads 

Doing this manually becomes difficult.

Kubernetes helps manage and orchestrate containers across infrastructure.

> **Docker = containerization**

> **Kubernetes = container orchestration**

### **For our project**

Kubernetes is something we may learn/use as part of the MLOps journey, but **it is not necessarily required for the first V1 implementation**.

We should first understand the simpler deployment architecture and introduce Kubernetes when there is an actual need for orchestration/scaling.

---

## **7. What does CI/CD solve?**

### **CI = Continuous Integration**

Continuous Integration means automatically integrating and validating code changes.

When a developer pushes changes, CI can automatically:

-  get the latest code 
-  install dependencies 
-  run tests 
-  check whether the application builds successfully 
-  catch problems early 

Basically:

> **"Did my new code break the project?"**

---

### **CD = Continuous Delivery / Continuous Deployment**

CD is about getting validated changes toward deployment.

Depending on the setup:

**Continuous Delivery:**

```
```

```
Code
 ↓
Test
 ↓
Build
 ↓
Ready for deployment
```

The deployment is prepared automatically, but a human may approve the final production deployment.

**Continuous Deployment:**

```
```

```
Code
 ↓
Test
 ↓
Build
 ↓
Deploy automatically
```

The validated change is automatically deployed.

### **Why CI/CD matters for our project**

Instead of manually repeating:

```
```

```
Change code
 ↓
Test
 ↓
Build
 ↓
Build Docker image
 ↓
Push image
 ↓
Deploy
```

we can automate much of this process.

---

## **8. What role does Google Cloud play?**

→ Google Cloud can provide cloud infrastructure and services needed to train, deploy, run, and monitor our ML application.

However:

> **The exact GCP services/configuration should be determined as we build rather than memorized upfront.**

### **Important for this project**

We should first understand **what infrastructure we actually need**, and then select the appropriate GCP services.

We should not add cloud services just because they sound impressive.

---

## **9. What role does GitHub play?**

→ GitHub hosts our project repository and allows us to maintain our source code, documentation, configuration, and project history using Git.

GitHub's major role here is:

> **Source-code hosting + version control + collaboration + project history**

And in our project, GitHub can also act as the source/trigger for our CI/CD pipeline.

For example:

```
```

```
You change code
      ↓
git add
      ↓
git commit
      ↓
git push
      ↓
GitHub
      ↓
CI/CD pipeline starts
```

### **Important correction**

GitHub does **not mean that we should upload everything** from the project.

For example, we generally should not blindly upload:

-  large raw datasets 
-  secrets/API keys 
- `.env` files 
-  virtual environments 
-  huge model weights 
-  unnecessary generated files 

Instead, GitHub should contain the important project code, documentation, configuration, tests, and appropriate references to external datasets/models.

---

## **10. What happens from** **`git push`** **to a user receiving a prediction?**

From the moment a Machine Learning Engineer performs `git push` to the point where an end-user receives a prediction, the project can involve two connected parts:

1. **CI/CD pipeline** 
2. **Production inference infrastructure** 

Imagine you change something in your project.

You do:

```
```

```
git add .
git commit
git push
```

Now your code reaches GitHub.

Without CI/CD, you might have to manually do:

```
```

```
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
```

That's annoying and, more importantly, easy to make mistakes.

With CI/CD, we want the system to automate these steps where appropriate.

### **Simplified pipeline**

```
```

```
                YOU
                 │
              git push
                 ↓
              GitHub
                 │
                 ↓
          CI/CD Pipeline
                 │
        ┌────────┴────────┐
        ↓                 ↓
      Build             Test
        │                 │
        └────────┬────────┘
                 ↓
        Build Docker image
                 ↓
        Push image to registry
                 ↓
          Deploy / update
                 ↓
        Running application
```

### **Then comes inference**

Once the application is deployed:

```
```

```
User
  ↓
Uploads image
  ↓
API
  ↓
Image preprocessing
  ↓
ML model
  ↓
Prediction
  ↓
Location + confidence
  ↓
User
```

So the complete high-level idea becomes:

```
```

```
Developer
   ↓
git push
   ↓
GitHub
   ↓
CI/CD
   ↓
Build + Test
   ↓
Docker image
   ↓
Deployment
   ↓
Running ML API
   ↑
   │
User ── image ──→ API
                   ↓
                 Model
                   ↓
              Prediction
                   ↓
              User result
```

# -[X] Phase 1 — Research & Requirements 🔎

> 1. Problem research ->> docs/research.md

> 2. Existing solutions ->> docs/research.md

> 3. Dataset research ->> docs/dataset-analysis.md

> 4. Define requirements ->> docs/requirements.md

> 5. Define project scope ->> docs/requirements.md

> 6. Technical decision ->> docs/technical-plan.md





