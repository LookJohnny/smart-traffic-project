<h1 align="center">🚦 Smart Traffic Management System 🚦</h1>
<h3 align="center">A scalable microservice-based application for predicting dangerous driving behaviors and detecting vehicle overloads in real-time</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Spring%20Boot-2.6.2-brightgreen" alt="Spring Boot">
  <img src="https://img.shields.io/badge/Kafka-Apache-yellowgreen" alt="Kafka">
  <img src="https://img.shields.io/badge/PyTorch-1.9.0-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/Docker-20.10-blue" alt="Docker">
  <img src="https://img.shields.io/badge/Redis-6.2.5-orange" alt="Redis">
</p>

## 📖 Overview
The **Smart Traffic Management System** is a comprehensive application designed to manage and analyze vehicle data streams to detect dangerous driving behaviors and overloads. It utilizes advanced technologies like **Spring Boot**, **Kafka**, **Redis**, and **PyTorch** to provide a real-time solution for traffic management.

### ✨ Features
- **Real-time Vehicle Data Processing**: Integration with **Kafka** to handle streaming data from vehicle sensors.
- **Driving Behavior Prediction**: Machine learning models (RNN/GRU) to predict behaviors such as aggressive driving.
- **Overload Detection**: Monitors sensor data to detect if vehicles are overloaded.
- **Redis Caching**: Low-latency data access using **Redis** for efficient performance.
- **Scalable Architecture**: Built using **Spring Boot**, making it easy to manage, scale, and deploy.

## 🏗️ Technology Stack
- **Backend Framework**: Spring Boot (Java)
- **Machine Learning**: PyTorch (Python)
- **Message Broker**: Apache Kafka
- **Data Caching**: Redis
- **Build Tool**: Maven
- **Deployment**: Docker & Flask (for Python ML service)

## 🚀 Getting Started
### Prerequisites
- Java 17
- Python 3.8+
- Maven
- Redis
- Kafka
- Docker

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/LookJohnny/smart-traffic-project.git
   cd smart-traffic-project

Technology Stack
Backend Framework: Spring Boot (Java)
Machine Learning: PyTorch (Python)
Message Broker: Apache Kafka
Data Caching: Redis
Build Tool: Maven
Model Type: RNN/GRU for time-series analysis
Deployment: Docker, Flask (for the Python ML service)
Architecture
Java Microservice: Handles Kafka integration, Redis caching, and business logic.
Python Microservice: Hosts the ML model for prediction and analysis, accessible via Flask APIs.
Kafka: Used for streaming sensor data, ensuring scalability and reliability in real-time environments.
Setup and Installation
Prerequisites
Java 17 (for Spring Boot backend)
Python 3.8+ (for ML model and Flask API)
Maven (for building the Java project)
Redis (for caching)
Kafka (for message streaming)
Docker (for containerization, optional)
Set up the Java backend:
bash
复制代码
mvn clean install
mvn spring-boot:run
Set up the Python Machine Learning service:
bash
复制代码
cd smart-traffic-ml
pip install -r requirements.txt
python app.py
Start Redis and Kafka using Docker:
bash
复制代码
docker run -p 6379:6379 redis
docker-compose up
🧪 Usage
Spring Boot Backend: Access it at http://localhost:8080.
Flask ML Service: Access it at http://localhost:5000/predict.
API Endpoints
POST /predict: Submit time-series sensor data for analysis and receive predictions.
GET /vehicles: Retrieve vehicle data stored in Redis.
🧠 Machine Learning Model
Model Architecture: GRU-based model (Gated Recurrent Unit) for processing time-series data from sensors such as accelerometers and gyroscopes.
Training Dataset: Trained on labeled data representing various driving behaviors (e.g., NORMAL, AGGRESSIVE).
Input: Sensor data (AccX, AccY, AccZ, GyroX, GyroY, GyroZ) collected over time windows.
Output: Probability of dangerous driving behavior or vehicle overload.
📈 Future Improvements
Model Optimization: Enhance accuracy with advanced techniques like hyperparameter tuning and feature engineering.
Scaling: Implement horizontal scaling using Kubernetes to handle increased traffic.
Edge Deployment: Deploy parts of the system on edge devices for faster real-time predictions.
🤝 Contributing
We welcome contributions! If you have any suggestions or want to collaborate, please submit a pull request or open an issue.

📫 Contact
Reach me at johnny.liu0888@gmail.com
Connect with me on LinkedIn
🛠️ Languages and Tools
<p align="left"> <a href="https://spring.io/projects/spring-boot" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/spring/spring-original-wordmark.svg" alt="Spring Boot" width="40" height="40"/> </a> <a href="https://kafka.apache.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/apachekafka/apachekafka-original-wordmark.svg" alt="Kafka" width="40" height="40"/> </a> <a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytorch/pytorch-original-wordmark.svg" alt="PyTorch" width="40" height="40"/> </a> <a href="https://www.docker.com/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="Docker" width="40" height="40"/> </a> <a href="https://redis.io/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/redis/redis-original-wordmark.svg" alt="Redis" width="40" height="40"/> </a> <a href="https://maven.apache.org/" target="_blank" rel="noreferrer"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/apache/apache-original-wordmark.svg" alt="Maven" width="40" height="40"/> </a> </p>
