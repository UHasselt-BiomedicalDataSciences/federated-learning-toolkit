---
title: The Journey of Data Within a Global Data Sharing Initiative - A Federated 3-Layer Data Analysis Pipeline to Scale Up Multiple Sclerosis Research
contributors: [Axel Faes]
page_id: gdsi
search_exclude: false
---

The paper "The Journey of Data Within a Global Data Sharing Initiative: A Federated 3-Layer Data Analysis Pipeline
 to Scale Up Multiple Sclerosis Research" describes an advanced method for collecting and analyzing data related
  to multiple sclerosis (MS) on a global scale {% cite Pirmani2023GDSI %}, [(external link)](https://medinform.jmir.org/2023/1/e48030/). Here's a simplified explanation:

## Overview 
### Short Summary

#### Overview

**Background**:
- Studying diseases like MS is difficult because not many people have them, and the data about these patients is scattered in different places and formats.
- This makes it hard to combine, standardize, and analyze the data effectively.

**Objective**:
- The paper presents a new system designed to collect and analyze data from various sources in a standardized way to improve research on MS.

#### Key Points

**1. Data Analysis Pipeline**:
- The pipeline is an organized process that takes raw data from multiple sources, improves its quality, integrates it into a single dataset, and analyzes it to produce meaningful results.
- This system was used successfully for COVID-19 and MS research.

**2. Three Data Sharing Streams**:
- **Direct Entry**: Patients and clinicians directly enter data into a central system using standardized forms.
- **Core Data Set Sharing**: Institutions share a subset of their data in a standardized format with a central platform.
- **Federated Model Sharing**: Data stays at the local institutions but is analyzed centrally using aggregated data to ensure privacy and security.

**3. Standardization and Quality Enhancement**:
- A data dictionary ensures that data from different sources is described in the same way, making it easier to integrate.
- Data is checked for quality, and any issues are flagged for correction.

**4. Integration and Analysis**:
- Data from the three streams is combined into one unified dataset.
- Advanced statistical methods are used to analyze this integrated data to find patterns and insights about MS and its interaction with COVID-19.

**5. Results**:
- The system successfully gathered the largest dataset of MS patients who contracted COVID-19 from 80 countries.
- The analysis helped understand the impact of various treatments on COVID-19 outcomes in MS patients.

#### Significance

**Global Collaboration**:
- The initiative shows how international collaboration and data sharing can lead to significant advancements in understanding and treating diseases.
- It emphasizes the importance of a well-organized data management system to handle real-world data.

**Challenges and Solutions**:
- Handling different data formats and quality levels is a major challenge.
- The paper suggests using standardized models and quality assessment frameworks to improve data integration and analysis.

**Privacy and Security**:
- Ensuring patient data privacy is crucial. The federated model helps by keeping detailed patient data at the local level while still allowing for centralized analysis.

#### Conclusion

The paper demonstrates a comprehensive method for managing and analyzing data on a global scale. This approach can significantly enhance research and provide valuable insights into diseases like MS, showing the power of collaborative data sharing initiatives.

## Reimplementation 

### Reimplementing the Work

To reimplement the work outlined in "The Journey of Data Within a Global Data Sharing Initiative: A Federated 3-Layer Data Analysis Pipeline to Scale Up Multiple Sclerosis Research," follow these detailed steps:

#### 1. Data Preparation

1. **Dataset Acquisition**:
   - Collect data from multiple sclerosis (MS) registries, ensuring compliance with ethical standards and data privacy regulations. Utilize sources such as direct entry from patients and clinicians, core data set sharing, and federated model sharing.

2. **Data Preprocessing**:
   - Standardize the data using a specialized data dictionary to ensure consistency across various data sources. This involves transforming and cleaning the data according to predefined schemas.
   - Ensure data integrity by removing duplicates, handling missing values, and validating data points.

3. **Data Quality Assessment**:
   - Implement a data quality assessment framework. Each data variable should be evaluated against binary criteria: PASS or FAIL. Variables failing the quality checks should be flagged for further inspection or correction.

#### 2. Data Acquisition Framework

1. **Direct Entry**:
   - Set up a web-based form for direct data entry by patients and clinicians. Ensure the form aligns with the data dictionary to facilitate seamless integration.
   - Implement strict privacy measures to exclude specific identifiers and prevent the use of cookies and trackers.

2. **Core Data Set Sharing**:
   - Develop secure interfaces for data providers to upload subsets of their datasets to the central platform. Standardize data formats according to the data dictionary before upload.
   - Enforce stringent data security measures, including user activity monitoring and access restrictions to ensure data confidentiality.

3. **Federated Model Sharing**:
   - Deploy Docker containers on each registry’s infrastructure to run scripts locally, standardize data, and compute aggregated data (buckets). Transfer the computed buckets to the central platform.
   - Conduct local quality checks before data aggregation to ensure high-quality data submissions.

#### 3. Data Integration

1. **Unified Data Structure**:
   - Integrate data from various sources into a unified dataset. Convert individual data points into a multivariate contingency table to facilitate downstream statistical analysis.
   - Aggregate data by adding patient counts for each variable combination across all data sources.

#### 4. Data Analysis

1. **Statistical Models**:
   - Employ multilevel mixed-effects logistic regression to analyze the aggregated data. This model assesses associations between disease-modifying therapies (DMTs) and COVID-19 severity outcomes, adjusting for variables such as age, sex, MS phenotype, and disability score.

2. **Evaluation Metrics**:
   - Use metrics such as hospitalization, intensive care unit admission, ventilation, and death to evaluate the impact of DMTs on COVID-19 severity.
   - Present findings using adjusted odds ratios (aOR) and confidence intervals (CI) to quantify the associations.

#### 5. Implementation Tools

1. **Programming Languages**: Python for scripting, data processing, and model implementation.
2. **Frameworks**:
   - **Docker**: For containerizing scripts and ensuring consistency across different environments.
   - **Pandas**: For data manipulation and preprocessing.
   - **Scikit-learn**: For implementing and evaluating machine learning models.
   - **Statsmodels**: For advanced statistical modeling.

3. **Version Control**:
   - Use Git and platforms like GitHub to manage code, documentation, and version control.

#### 6. Deployment

1. **Server and Client Setup**:
   - Set up a centralized server to coordinate data collection, integration, and analysis.
   - Deploy client interfaces for data providers to upload and manage their data contributions securely.

2. **Documentation and User Support**:
   - Provide detailed documentation, including setup guides, user manuals, and illustrative visuals to assist users in navigating the data analysis pipeline.
   - Develop a user-centric interactive web application to enhance user experience and facilitate collaboration.

#### 7. Continuous Improvement

1. **Stakeholder Engagement**:
   - Engage with stakeholders regularly to gather feedback and improve the data analysis pipeline.
   - Foster collaboration through workshops, webinars, and community forums.

2. **Privacy and Compliance**:
   - Ensure ongoing compliance with data privacy regulations by conducting regular audits and updates to security protocols.
   - Implement privacy-preserving algorithms, such as differential privacy and homomorphic encryption, to enhance data security.

By following these steps, you can successfully reimplement the federated 3-layer data analysis pipeline described in the paper and leverage it for comprehensive multiple sclerosis research using real-world data. The integration of advanced data management and federated learning techniques ensures scalability, flexibility, and compliance with privacy standards, making it a robust framework for collaborative healthcare research.

## Results

The implementation of the Global Data Sharing Initiative (GDSI) provided significant insights and outcomes in multiple sclerosis (MS) research and the impact of COVID-19 on people with MS. Here are the key results:

1. **Data Acquisition**:
   - The GDSI successfully assembled the largest cohort of people with MS infected with COVID-19. Data were collected from 80 countries, with significant contributions from the United States, Australia, Spain, Sweden, Germany, Argentina, Brazil, Turkey, Denmark, and the United Kingdom.
   - The initiative utilized three distinct data sharing streams: direct entry, core data set sharing, and federated model sharing. This hybrid approach allowed for comprehensive data collection despite varying degrees of data-sharing willingness and regulatory constraints.

2. **Data Analysis**:
   - Analysis focused on assessing the impact of different disease-modifying therapies (DMTs) on COVID-19 severity among people with MS. The study employed multilevel mixed-effects logistic regression to analyze variables such as age, sex, MS phenotype, disability score, DMTs, and COVID-19 severity.
   - Key findings included higher risks of hospitalization, intensive care unit admission, and artificial ventilation for patients using rituximab and ocrelizumab compared to other DMTs. However, neither rituximab nor ocrelizumab was significantly associated with an increased risk of death.

### Discussion

The discussion of the results highlights several key insights and challenges:

1. **Insights from the GDSI Study on MS and COVID-19**:
   - The GDSI provided invaluable data for understanding the impact of COVID-19 on people with MS, emphasizing the importance of evidence-based decision-making in disease management. The collaborative approach of involving neurologists, patients, and registries worldwide was crucial for the success of the initiative.
   - Observational studies, such as this one, offer significant real-world insights, although they come with inherent limitations. The study's design and execution provide a model for future large-scale, international collaborative research efforts.

2. **Challenges and Solutions in Data Interoperability, Quality, and Governance**:
   - Interoperability and handling heterogeneous data formats were major challenges. The creation of a study-specific data dictionary helped mitigate these issues. However, adopting more advanced standardization methods, such as common data models and frameworks like Fast Healthcare Interoperability Resources (FHIR), could further enhance data integration and analysis.
   - Ensuring data quality was paramount. GDSI implemented an automated data quality assessment framework, but further improvement could be achieved by adopting generalized frameworks for data quality assessment across various healthcare contexts.
   - Navigating regulatory compliance and data governance in a federated framework posed significant challenges. Implementing a federated governance model helped address these issues, but the need for a more universal data governance model remains.

3. **Embracing Federated Model Sharing and Privacy Concerns**:
   - Federated model sharing allowed insights to be drawn from patient-level data without transferring raw data, mitigating privacy risks. However, even aggregated statistics can pose privacy concerns. GDSI's rigorous privacy assessments and transparent communication with data providers helped manage these risks.
   - Federated learning, which allows machine learning algorithms to learn from distributed data without centralizing it, offers a promising solution but comes with its own set of challenges and risks. Incorporating privacy-preserving algorithms like differential privacy and homomorphic encryption can enhance security, though they may impact analytical performance.

4. **Enhancing Collaboration and User Engagement**:
   - Improving user experience and engagement was essential for the pipeline's success. GDSI addressed this by developing a user-centric interactive web application, detailed documentation, and illustrative visuals to demystify the pipeline's complexity.
   - Continuous education, proactive stakeholder engagement, and evidence-based demonstrations were crucial for fostering trust and collaboration among diverse stakeholders.

### Conclusion

The GDSI made substantial contributions to MS research and set a new standard for global data-sharing initiatives. The initiative demonstrated the potential of collaborative, data-driven approaches to improve healthcare outcomes and provided a scalable framework that can be adapted to other healthcare sectors. The hybrid approach to data acquisition and analysis, coupled with a strong emphasis on data quality, interoperability, and privacy, underscores the importance of comprehensive and flexible data management strategies in biomedical research.

## Bibliography

{% bibliography --cited %}
