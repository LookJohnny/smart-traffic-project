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
- **Backend Framework**: [Spring Boot (Java)](https://spring.io/projects/spring-boot)
- **Machine Learning**: [PyTorch (Python)](https://pytorch.org/)
- **Message Broker**: [Apache Kafka](https://kafka.apache.org/)
- **Data Caching**: [Redis](https://redis.io/)
- **Build Tool**: [Maven](https://maven.apache.org/)
- **Model Type**: RNN/GRU for time-series analysis
- **Deployment**: Docker, Flask (for the Python ML service)

### Architecture
- **Java Microservice**: Handles Kafka integration, Redis caching, and business logic.
- **Python Microservice**: Hosts the ML model for prediction and analysis, accessible via Flask APIs.
- **Kafka**: Used for streaming sensor data, ensuring scalability and reliability in real-time environments.

## 🚀 Setup and Installation

### Prerequisites
- **Java 17** (for Spring Boot backend)
- **Python 3.8+** (for ML model and Flask API)
- **Maven** (for building the Java project)
- **Redis** (for caching)
- **Kafka** (for message streaming)
- **Docker** (for containerization, optional)

### Steps to Setup

# Steps to Setup

# 1. Set up the Java backend:
mvn clean install
mvn spring-boot:run

# 2. Set up the Python Machine Learning service:
cd smart-traffic-ml
pip install -r requirements.txt
python app.py

# 3. Start Redis and Kafka using Docker:
docker run -p 6379:6379 redis
docker-compose up


📫 Contact
Reach me at johnny.liu0888@gmail.com
Connect with me on LinkedIn
