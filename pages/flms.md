---
title: Transforming Multiple Sclerosis Research - Pioneering Practical and Precise Approaches with Federated Learning using Real-World Data
contributors: [Axel Faes]
page_id: flms
search_exclude: false
---

The paper titled "Transforming Multiple Sclerosis Research: Pioneering Practical and Precise Approaches with Federated Learning using Real-World Data" focuses on the application of Federated Learning (FL) to Multiple Sclerosis (MS) research using real-world clinical data. Here's a simplified explanation of the key points and findings:

## Overview 
### Short Summary

#### What is the Study About?

- **Purpose**: To explore how Federated Learning (FL) can improve MS research using routine clinical data, and to identify the best FL configurations for analyzing MS data.
- **Goals**: To evaluate the effectiveness of FL in handling large-scale data sharing while maintaining data privacy, and to provide recommendations for data custodians.

#### Key Concepts

1. **Multiple Sclerosis (MS)**: A complex neurological disorder with varying symptoms and progression, making it challenging to manage and predict.
2. **Machine Learning (ML)**: A technology that can process large datasets to uncover insights about MS, such as disease progression and patient outcomes.
3. **Federated Learning (FL)**: A decentralized approach to ML that allows multiple institutions to collaboratively train a model without sharing their raw data, thus preserving privacy.

#### Methods Used

- **Data Partitioning**: The data was divided in different ways:
  - **Natural Partitioning**: Based on countries or clinics.
  - **Artificial Partitioning**: Using a method called Optimal Transport Dataset Distance (OTDD) to group similar datasets.
  - **Skewed Partitioning**: Dividing data into high, medium, and low skewed sets.
- **Federated Learning Configurations**: Different FL strategies were tested, including:
  - **FedAVG**: A basic averaging method.
  - **FedOpts**: Including variants like FedYogi, FedAdam, and FedAdagrad.
  - **FedProx**: Another FL strategy.

#### Key Findings

1. **Best Performing FL Model**: The OTDD approach using FedAVG achieved the highest performance, better than other federated models and even some centralized models.
2. **Country-Based Partitioning**: The country-based partitioning strategy using FedAVG was also effective, though not as good as OTDD.
3. **Client Participation**: Adjusting the number of clients participating in the training process can significantly reduce the experiment time without major performance loss.
4. **Fine-Tuning**: Fine-tuning federated models improved their performance significantly, especially in most countries tested.
5. **Trade-Offs**: There are trade-offs between performance and computational efficiency, with different configurations balancing these aspects differently.

#### Conclusion

The study demonstrates that FL can be a powerful tool for MS research, offering a way to handle decentralized, privacy-sensitive data effectively. It highlights the importance of choosing the right FL configurations to achieve optimal performance and suggests that FL has broader potential in medical research beyond just MS.

The full source code for this study is available at [GitHub](https://github.com/ashkan-pirmani/FL-MS-RWD), allowing for further exploration and replication of the study's findings.

## Reimplementation 

### Reimplementing the Work

To reimplement the work outlined in the paper, follow these detailed steps, covering both the data preparation and the federated learning setup:

#### 1. Data Preparation

1. **Dataset Acquisition**:
   - Collect data from multiple sclerosis (MS) registries, ensuring it adheres to the criteria specified in the study: minimum follow-up period of 12 months, age over 18 years, and diagnosis of MS or clinically isolated syndrome (CIS).

2. **Data Preprocessing**:
   - Ensure data integrity by removing duplicate or inconsistent records.
   - Segment each patient's clinical trajectory into episodes, each encapsulating an observation window, baseline Expanded Disability Status Scale (EDSS) score, and a confirmation label.
   - Ensure a minimum of three EDSS measurements per patient.
   - Finalize the dataset with quality-assured data points ready for analysis.

3. **Partitioning**:
   - **Country-Based Partitioning**: Segment data into subsets based on the geographical origin (country). Each subset should contain at least 5 data points.
   - **Clinic-Based Partitioning**: Segment data based on the medical clinic associated with each data point.
   - **Similarity-Based Partitioning**: Use Optimal Transport Dataset Distance (OTDD) to cluster similar datasets. Create a distance matrix and apply hierarchical clustering to group countries into coherent clusters.
   - **Quantity-Based Partitioning**: Implement heuristic random partitioning schemes with high, mid, and low skewness levels to distribute data into various quantity-based subsets.

#### 2. Federated Learning Setup

1. **Model Architecture**:
   - Use a Multi-Layer Perceptron (MLP) model with five hidden layers, each containing 256 neurons, and an input layer with 42 neurons.

2. **Experimental Setup**:
   - **Federated Model Training**:
     - Simulate a server-client architecture where the server coordinates the learning process, and clients participate in distributed training.
     - Initialize the model on the server and distribute it to clients.
     - Each client trains the model locally on their partitioned dataset for E epochs.
     - Clients send their updated models and metrics back to the server.
     - The server performs federated aggregation to update the global model.
     - Iterate the process for N federation rounds.
   - **Configurations**:
     - Test different configurations of client participation (e.g., 100%, 60%, and 40% of clients).
     - Employ various federated strategies like FedAvg, FedAdam, FedAdagrad, and FedProx.

3. **Fine-Tuning**:
   - After federated training, each client fine-tunes the globally trained model on their local data to improve performance.
   - This step is crucial for adapting the model to the specific characteristics of each client’s data.

4. **Centralized and Local Models**:
   - Train a centralized model using the pooled global dataset as a benchmark.
   - Train local models independently on each client’s dataset to compare performance with federated models.

5. **Evaluation**:
   - Use a consistent test dataset partitioned by country to evaluate all models.
   - Metrics: Evaluate models using ROC-AUC, AUC-PR, and experiment time.
   - Perform extensive hyperparameter tuning and repeat experiments multiple times to ensure reliability.

6. **Recommendation Model**:
   - Conduct a comparative analysis to identify the best-performing model for each country.
   - Generate guidelines based on the performance metrics to help data custodians select optimal FL configurations for their specific needs.

### Tools and Resources

- **Programming Languages**: Python for scripting and model implementation.
- **Frameworks**:
  - **Flower (FL)**: A federated learning framework to manage server-client architecture and federated strategies.
  - **Machine Learning Libraries**: TensorFlow or PyTorch for implementing the MLP model.
- **Data Processing**: Pandas for data manipulation, NumPy for numerical operations, and Scikit-learn for preprocessing and evaluation metrics.
- **Hyperparameter Tuning**: Use GridSearchCV or similar tools to optimize model parameters.

### Step-by-Step Guide

1. **Set Up Environment**:
   - Install required libraries and frameworks (Flower, TensorFlow/PyTorch, Pandas, NumPy, Scikit-learn).

2. **Prepare Data**:
   - Load and preprocess data according to the steps mentioned above.
   - Partition data using the specified strategies.

3. **Implement Federated Learning**:
   - Initialize the federated learning setup with the Flower framework.
   - Define the MLP model architecture.
   - Configure and run federated training experiments with various client fractions and federated strategies.
   - Implement fine-tuning for the trained federated models.

4. **Evaluate Models**:
   - Use consistent test datasets to evaluate federated, centralized, and local models.
   - Record performance metrics and experiment times.

5. **Analyze Results**:
   - Perform a comparative analysis of the models to determine the best-performing configurations.
   - Generate guidelines and recommendations based on the results.

6. **Document and Share**:
   - Document the entire process, including code, configurations, and results.
   - Share the implementation on platforms like GitHub for community use and further research.

By following these detailed steps, you can successfully reimplement the work outlined in the paper and leverage federated learning for multiple sclerosis research using real-world data.

## Results

1. **Federated Model Performance**:
   - Various federated learning (FL) configurations were tested, identified by mnemonics such as 'Country AVG', 'Country 60% AVG', 'OTDD AVG', 'HighSkewed AVG', etc.
   - Extensive hyperparameter tuning was conducted for the 'Country AVG' model to ensure consistency across all experiments. Each experiment was repeated five times to ensure reliability.
   - **Performance Metrics**: 
     - The models were evaluated using the Receiver Operating Characteristic Area Under the Curve (ROC-AUC) and the Area Under the Precision-Recall Curve (AUC-PR).
     - **Experiment Time**: Total time taken for training from the beginning to the end of the last federation round.

   - **Key Findings**:
     - The OTDD clustering and heuristic random partitioning within the FL context significantly improved model performance.
     - The SoloSync model outperformed others with a ROC-AUC of 0.7970 and AUC-PR of 0.3822, taking 48.29 minutes on average per experiment.
     - Centralized models, used as benchmarks, showed lower performance (ROC-AUC: 0.7423, AUC-PR: 0.2930) but required less time (9.12 minutes on average).

2. **Comparative Analysis**:
   - Different partitioning strategies and federated algorithms were compared, highlighting the following:
     - **Country-Based Partitioning**: 
       - 'Country AVG' achieved a ROC-AUC of 0.7295 and AUC-PR of 0.2767, with an average experiment time of 28.93 minutes.
       - 'Country Adagrad' and 'Country Adam' showed slightly lower performance but required longer experiment times.
     - **Clinic-Based Partitioning**:
       - 'Clinic AVG' and 'Clinic Prox' showed similar performance with a ROC-AUC around 0.7219 and AUC-PR of 0.2802 but had significantly longer experiment times (81.95 and 92.17 minutes, respectively).

3. **Model Robustness**:
   - The federated models demonstrated robustness in handling data heterogeneity and non-IID (non-independent and identically distributed) data scenarios.
   - The consistent use of a test dataset partitioned by country (DCi) ensured a uniform evaluative framework across all experiments, preventing bias and ensuring reliability.

### Discussion

1. **Advantages of Federated Learning**:
   - FL enables training on distributed datasets without the need to centralize data, thus preserving privacy and complying with data regulations.
   - The flexibility of FL allows for various data partitioning strategies tailored to specific national or clinical contexts, enhancing the model's applicability.

2. **Challenges and Limitations**:
   - The dispersion of data and the necessity for multiple communication rounds introduce computational overhead and longer training times.
   - The performance of federated models often falls short of centralized models, particularly in scenarios involving numerous clients with limited data.
   - Fine-tuning models to specific datasets emerged as a critical factor for achieving optimal performance, challenging the traditional emphasis on large datasets.

3. **Practical Implications**:
   - The findings emphasize the importance of balancing the number of clients and data volume per client to optimize performance and efficiency.
   - Country-specific and clinic-specific guidelines derived from the experiments can inform practical implementations of FL in real-world settings, such as multiple sclerosis research using real-world data (RWD).

4. **Future Work**:
   - Further research is needed to explore more sophisticated partitioning strategies and optimization techniques to reduce computational overhead.
   - Investigating the impact of different types of data heterogeneity on model performance can provide deeper insights into the strengths and limitations of FL in various contexts.

By providing a comprehensive evaluation of different FL configurations and strategies, this paper highlights the potential of federated learning to enhance collaborative research while addressing data privacy and regulatory challenges.
