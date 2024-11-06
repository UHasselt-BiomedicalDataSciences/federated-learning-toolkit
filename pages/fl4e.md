---
title: Federated Learning for Everyone (FL4E)
---

The paper "Federated Learning for Everyone (FL4E)" presents a new approach to improve collaboration in clinical research using federated learning. Here's a simplified explanation:

## Overview 
### Short Summary

#### Background
Clinical research relies heavily on vast amounts of data from multiple sources, such as hospitals and clinics. However, accessing and sharing this data is challenging due to privacy concerns, ethical issues, and regulatory barriers. Traditional methods of data sharing often require centralizing the data, which can be impractical or even impossible due to these constraints.

#### What is Federated Learning?
Federated learning (FL) is a method that allows machine learning models to be trained on data distributed across different locations without the need to centralize the data. Instead, the model is trained locally on each dataset, and only the trained models (not the data itself) are shared and combined to create a global model.

#### Objectives of FL4E
The FL4E framework aims to make federated learning more accessible and practical for various stakeholders in the healthcare industry. It addresses several key challenges:
1. **Implementation Complexity:** Simplifies the setup and use of federated learning systems.
2. **Scalability:** Ensures the system can handle large-scale data and numerous participants.
3. **Inclusivity:** Makes the system adaptable to different levels of data sharing and privacy requirements.

#### Key Concepts of FL4E
1. **Degree of Federation:** This concept allows for flexibility in how much data is centralized versus kept local. It means stakeholders can choose a balance that best suits their privacy and operational needs. For instance, some participants might share data centrally while others use a federated approach.
2. **Ecosystem-Based Collaboration:** FL4E promotes a collaborative environment where various stakeholders (researchers, healthcare providers, patients, etc.) can work together more effectively. This holistic approach integrates diverse expertise and resources to enhance research outcomes.

#### Results
The paper evaluates FL4E using real-world healthcare data and shows that hybrid models, which combine elements of both centralized and federated learning, can achieve performance similar to fully federated models but with less complexity and overhead. This makes it easier for diverse stakeholders to participate and benefit from shared insights.

#### Conclusion
FL4E represents a significant advancement in collaborative clinical research by offering a flexible and inclusive framework. Its design allows for various levels of data sharing and protection, making it suitable for a wide range of research scenarios. This approach promises to enhance collaboration and data utilization in healthcare, ultimately leading to better research and patient outcomes.

In summary, FL4E simplifies federated learning, making it more accessible and adaptable, thereby fostering better collaboration and more efficient use of data in clinical research.

### FL4E Server Structure

The FL4E server is designed to manage the overall process of federated learning while ensuring that data remains secure and private. Here's a step-by-step explanation of how the server is set up and functions:

1. **Microservice-Based Web Application:**
   - The server is built as a microservice-based web application. This means it is divided into small, independent services that work together. Each service handles a specific part of the system's functionality, making it easier to manage and scale.

2. **Technological Stack:**
   - **ASP.NET Core:** The server uses ASP.NET Core, a framework for building web applications. This framework is known for its performance and ease of use.
   - **Docker:** Both server and client components are containerized using Docker. This ensures that the application runs consistently across different environments and makes deployment easier.

3. **Architecture Layers:**
   - The server application is divided into several layers, each responsible for different tasks:
     - **Application Layer:** Manages the overall logic of the application, including user authentication and authorization.
     - **Model Layer:** Defines the structure of the data used in the application.
     - **Utility Layer:** Handles common functions and utilities used by other layers.
     - **Data Access Layer:** Interacts with the database to store and retrieve data.

4. **User Interaction:**
   - **Admin Control Panel:** An admin control panel is included for managing user registrations and overseeing the platform.
   - **Web Application:** The primary user interface is a web application that users interact with to initiate and manage federated learning tasks.

5. **Core Components:**
   - **Server Container:** The main server component runs as a Docker container. This container includes:
     - A web application built with ASP.NET.
     - An SQL server for data storage.
     - A Python environment for handling data processing and federated learning tasks.
   - **Client Containers:** These are Docker containers that run on the data providers' machines. They are used to execute federated learning scripts locally on the data providers' datasets.

6. **Communication and Data Flow:**
   - The server acts as the coordinator for federated learning tasks. It does not directly handle raw data from clients. Instead, it manages the sharing and updating of models trained on local data at each client site.
   - **Flower Framework:** The backend uses the Flower framework to manage federated learning processes. Flower is flexible and supports multiple machine learning frameworks and programming languages.
   - **Security:** Secure channels are used for all communications between the server and clients to ensure data privacy and integrity.

7. **Operational Flow:**
   - **Initiating Studies:** Data scientists can create and manage study entries in a "Study Center" on the server.
   - **Sharing Analysis Materials:** Scripts for federated learning are shared through the server, and data providers run these scripts locally using the client component.
   - **Aggregation of Results:** After local training, the results (model updates) are sent back to the server, which aggregates them to update the global model.

8. **Deployment and Accessibility:**
   - The server can be deployed to cloud platforms such as Microsoft Azure, making it accessible for users without complex infrastructure setup.
   - **GitHub Repository:** The source code for both server and client components is available on GitHub, promoting transparency and community contributions.

### Simplified Use Case
- **Step 1:** A data scientist initiates a study on the server, providing necessary scripts.
- **Step 2:** Data providers review the study and agree to participate.
- **Step 3:** Data providers run the provided scripts locally to train models on their data.
- **Step 4:** The local updates are sent back to the server, which aggregates them to improve the global model.
- **Step 5:** The updated global model can be shared back with all participants for further use or validation.

This structure ensures that federated learning can be performed efficiently and securely, allowing multiple stakeholders to collaborate without compromising their data privacy.

### Main modules

Here are the main modules and their functionalities:

1. **Study Center**:
   - **Purpose**: Coordinates research activities and serves as a cooperative repository for stakeholders to share and access information or metadata about various studies.
   - **Components**:
     - **Study Catalogue**: A repository where stakeholders can initiate collaborations by sharing overarching information or metadata about different studies. It facilitates the pooling of resources and expertise, enhancing the collaborative research environment.
     - **Analysis Center**: Primarily designed for data scientists to create detailed analysis entries derived from study catalogue records. This includes providing titles, descriptive summaries, and privacy settings for analyses, ensuring only authorized users can access certain data entries.

2. **Repository Center**:
   - **Purpose**: Acts as a warehouse for all scripts and models essential for research and collaboration, facilitating dissemination and archiving.
   - **Functions**: Integrates with the Study Center and Model Center to provide organized access to shared scripts and models. It stores past analyses and provides examples for creating new studies.

3. **Model Center**:
   - **Purpose**: Facilitates the coordination and sharing of trained models.
   - **Components**:
     - **Model Catalogue**: A database for high-level information about various trained models, including details such as title, description, type of model, and status.
     - **Model Repository**: Stores detailed information about each trained model, ensuring only authorized users can access certain models. It can also disseminate models more broadly if needed, allowing users to validate models against their datasets or use them as pre-trained models.

4. **Data Center**:
   - **Purpose**: Central to the concept of the "degree of federation" within the FL4E framework, providing data sharing, integration, and aggregation capabilities.
   - **Functions**: 
     - **Raw Data Sharing Scheme**: Initiates the process when the data provider shares a raw data scheme via a data dictionary.
     - **Data Cleaning and Enhancement Script**: Ensures data quality by handling duplicates, missing values, and validating value ranges.
     - **Data Sharing and Aggregation**: Facilitates data sharing, triggering cleaning scripts and ensuring data is aggregated into a single file for analytical use .

Each module is designed to enhance the collaborative environment within the FL4E framework, ensuring secure, efficient, and organized data sharing and analysis, tailored to the needs of various stakeholders in healthcare research.

### Key Concepts of FL4E

The key concepts of the FL4E framework are designed to facilitate adaptable and inclusive federated learning (FL) in healthcare research. Here are the main concepts:

1. **Degree of Federation**:
   - **Concept**: This is a central idea in FL4E, offering a flexible approach to collaborative data analysis by allowing participants to choose their level of involvement and data-sharing based on their specific needs and constraints.
   - **Customization**: Stakeholders can select between centralized and federated models based on their infrastructural capacities, regulatory requirements, and privacy considerations. This continuum is not a strict dichotomy but a spectrum where stakeholders can balance operational efficiency and data privacy risk .
   - **Applications**: It accommodates different participation levels, making FL more practical and ensuring secure data management even in semi-centralized scenarios. This flexibility has proven effective in initiatives like the Global Data Sharing Initiative for Multiple Sclerosis and COVID-19, enhancing data collection and inclusivity .

2. **Ecosystem-Based Collaborative Learning**:
   - **Concept**: FL4E shifts from traditional project-specific FL to an "ecosystem-based" approach, creating a unified, adaptable platform for conducting various analyses focused on specific diseases or health conditions.
   - **Holistic Perspective**: This methodology aims to form a comprehensive, synergistic environment for data sharing and collaborative research. It includes a diverse network of stakeholders, such as researchers, healthcare professionals, patients, and policymakers, united by common research goals .
   - **Adaptability**: The ecosystem approach is designed to be adaptable, evolving with new research findings, emerging health challenges, and regulatory changes, ensuring long-term sustainability and relevance .

3. **Foundational Principles**:
   - **Flower Framework**: FL4E builds on the Flower framework, which supports multi-node execution, is agnostic to machine learning frameworks and programming languages, and enhances FL4E's flexibility and versatility .
   - **Modular Design**: The framework is divided into four main centers (Study Center, Repository Center, Model Center, and Data Center), each serving specific roles in facilitating collaborative research and data sharing. This modularity allows stakeholders to engage selectively in relevant parts of the data-sharing pipeline .

These key concepts underscore FL4E's commitment to creating an inclusive, adaptable, and secure environment for collaborative healthcare research, leveraging the strengths of both federated and centralized models to meet the diverse needs of stakeholders.

## Reimplementation 
To reimplement the work outlined in the FL4E framework, follow these detailed steps:

### Server-Side Implementation

1. **Framework and Architecture**:
   - The server-side is a microservice-based web application developed using ASP.NET Core.
   - It employs an N-tier architecture with the Model-View-Controller (MVC) design pattern and a repository pattern for scalability and maintainability.

2. **Components**:
   - **Layers**: The application is segmented into four layers:
     - **Application Layer**: Manages user authorization and the application's core logic.
     - **Model Layer**: Defines the data structures.
     - **Utility Layer**: Handles common utilities and helpers.
     - **Data Access Layer**: Interacts with the database.

3. **Authentication and Authorization**:
   - Uses the ASP.NET Core Identity module for managing user authentication and authorization.
   - An admin control panel supervises user registrations.

4. **Deployment**:
   - The server component is encapsulated as a Docker container, which facilitates ease of deployment and consistency across different environments.
   - A fully functional prototype is deployed on Microsoft Azure, showcasing the platform's robustness.

### Client-Side Implementation

1. **Framework and Interfaces**:
   - The client-side also uses the ASP.NET Core framework.
   - It includes three primary interfaces:
     - **Data Mounting Interface**: Allows users to select and mount data files.
     - **Script Uploading Interface**: Enables users to upload scripts obtained from the server.
     - **Script Execution Interface**: Allows running these scripts using the Flower framework to secure communication between the client and server.

2. **Execution Environment**:
   - The client component is also encapsulated as a Docker container.
   - This ensures portability and ease of deployment.

### Essential Tools and Technologies

1. **Flower Framework**:
   - Supports multi-node execution and is agnostic to machine learning frameworks and programming languages.
   - Provides flexibility and versatility for federated learning tasks.

2. **Docker**:
   - Used to encapsulate both client and server components, ensuring consistent environments and simplified deployment processes.

### Steps for Reimplementation

1. **Set Up Development Environment**:
   - Install Docker and Docker Compose.
   - Set up an ASP.NET Core development environment.

2. **Clone the Repository**:
   - Access the source code for both the server and client components from the GitHub repository: [FL4E-Analysis](https://github.com/ashkan-pirmani/FL4E-Analysis).

3. **Configure Server-Side Components**:
   - Define data structures in the Model Layer.
   - Implement user authorization and authentication in the Application Layer.
   - Set up database interactions in the Data Access Layer.
   - Implement the MVC pattern for handling user requests and responses.

4. **Configure Client-Side Components**:
   - Develop interfaces for data mounting, script uploading, and script execution.
   - Ensure secure communication using the Flower framework.

5. **Deploy Using Docker**:
   - Write Dockerfiles for both server and client components.
   - Use Docker Compose to manage multi-container applications.
   - Deploy the Docker containers to a cloud service like Microsoft Azure.

6. **Run and Test the System**:
   - Mount data files using the client interface.
   - Upload and execute scripts to ensure proper functionality.
   - Test the end-to-end workflow from data mounting to analysis execution.

7. **Documentation and Tutorials**:
   - Refer to the comprehensive video demonstration available to understand the platform's operational flow and features.
   - Use the provided foundational scripts in the GitHub repository as guidelines for your experiments.

### Additional Resources

- **Documentation**: Ensure detailed documentation is provided for each component and interface.
- **Community Support**: Engage with the community for continuous improvement and collaborative development.
- **Privacy and Compliance**: Ensure adherence to data privacy regulations and secure data management practices.

By following these detailed steps, you can successfully reimplement the FL4E framework and leverage its capabilities for federated learning in healthcare research     .

## Results

The FL4E framework was evaluated using two real-world clinical datasets: Fed-Heart-Disease and Fed-Tcga-Brca. Here are the key results:

1. **Datasets and Models**:
   - **Fed-Heart-Disease**: Contains 740 records from four centers with 13 clinical features and a binary heart disease indicator. A logistic regression model was used for prediction.
   - **Fed-Tcga-Brca**: Contains data from 1,088 breast cancer patients across six centers with 39 clinical features and each patient's time of death. A Cox model was used to predict the risk of death.
   - **Wisconsin Breast Cancer Dataset**: Included in the GitHub repository as an extra use case to provide foundational scripts for hybrid experiment settings in the FL4E framework.

2. **Degree of Federation Scenarios**:
   - **Fully Federated Experiment**: Each client participated in the FL process, contributing to the global model with their local datasets. This scenario aligns with the conventional FL setup.
   - **Hybrid Experiment**: Some clients contributed their data centrally to the server while others participated as federated clients. For example, in the Fed-Heart-Disease dataset, Clients 1 and 2 contributed centrally, while others participated as federated clients.
   - **Centralized Experiment**: All clients sent their data to the central server, where the model was trained centrally. This design aimed to benchmark the performance of federated and hybrid approaches against the centralized model.

3. **Performance Metrics**:
   - Evaluations were based on ROC-AUC and accuracy for the Fed-Heart-Disease dataset and the concordance index (C-index) for the Fed-Tcga-Brca dataset.
   - Multiple FL strategies were implemented: FedAVG, FedOpt (including FedAdagrad and FedYogi), and FedProx. Hyperparameter tuning was performed using grid search on a centralized setting and applied to each experiment. Each experiment was repeated five times for consistency.

4. **Results Summary**:
   - For the Fed-Heart-Disease dataset, the ROC-AUC and accuracy were highest in the fully federated setup using FedProx (ROC-AUC: 0.846 ± 0.003, Accuracy: 0.741 ± 0.006).
   - For the Fed-Tcga-Brca dataset, FedAdagrad in the hybrid setup performed well (C-Index: 0.776 ± 0.036), indicating that hybrid models can be competitive with fully federated models.

### Discussion

The results highlight several key points:

1. **Efficacy of Federated Learning**:
   - Federated learning, particularly in a fully federated setup, demonstrated robust performance, suggesting that FL models can leverage distributed datasets effectively to produce reliable predictive models.
   - Hybrid experiments, where some data is centralized while other data remains federated, showed comparable performance to fully federated setups. This suggests that hybrid models can balance operational efficiency and data privacy concerns while maintaining high performance.

2. **Impact of Data Imbalance**:
   - Smaller clients with imbalanced datasets benefited more from participating in an FL environment than operating in isolation. This insight is aligned with the foundational principles of FL, which aim to enhance model robustness and generalizability by harnessing diverse data sources.

3. **Centralized vs. Federated Models**:
   - Contrary to initial expectations that centralized models would outperform others, the analysis showed that federated and hybrid models exhibit superior performance based on ROC-AUC measures. This finding emphasizes the importance of evaluating FL models using a separate global test set to enhance the understanding of model performance across varied environments.

4. **Regulatory and Compliance Considerations**:
   - The flexible approach of FL4E, which allows varying degrees of federation, ensures compliance with data privacy regulations while providing operational efficiency. This adaptability makes FL4E suitable for a wide range of clinical research scenarios, from highly regulated environments to more flexible settings.

Overall, the FL4E framework demonstrates significant potential in enhancing collaborative clinical research through its flexible, ecosystem-based approach to federated learning. Its ability to balance centralized and federated learning models while maintaining high performance and regulatory compliance highlights its practical value in real-world healthcare settings.