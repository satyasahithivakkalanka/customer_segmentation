# Customer Segmentation Using K-Means

## Overview

This project performs customer segmentation based on *annual income*
and *spending score* using the K-Means clustering algorithm. The goal
is to identify distinct customer groups that can help a business tailor
marketing strategies and improve decision-making.

## Steps in the Project

1.  *Data Import and Cleaning* The dataset ⁠ Mall_Customers.csv ⁠ is
    loaded and cleaned. Only two key variables are used for clustering:

    -   Annual Income (k\$)
    -   Spending Score (1-100) There are no missing values in these
        columns.

2.  *Exploratory Data Analysis* Two histograms show the feature
    distributions:

    -   *Annual Income Distribution:* Most customers earn between 40k
        and 80k, with a smaller group earning above 100k.
    -   *Spending Score Distribution:* The scores are spread across
        the range, with visible clusters of high and low spenders.

3.  *Feature Scaling* Since income and spending score are measured on
    different scales, both features are standardized before clustering.

4.  *Choosing the Optimal Number of Clusters*

    -   *Elbow Method:* The curve starts to flatten around **k = 5 or
        6**, suggesting that this range captures most of the structure
        in the data.
    -   *Silhouette Score:* The highest score appears at *k = 5*,
        and scores remain strong up to *k = 6*, confirming a similar
        choice.

5.  *Model Training* The final K-Means model is trained with **6
    clusters** to balance accuracy and interpretability.

6.  *Cluster Centers* The cluster centers in original units are:

      Cluster     Annual Income (k\$)   Spending Score (1-100)
      --------- --------------------- ------------------------
      0                         55.30                    49.52
      1                         88.20                    17.11
      2                        109.70                    82.00
      3                         26.30                    20.91
      4                         25.73                    79.36
      5                         78.55                    82.17

    These centers represent the typical income and spending score for
    each segment.

7.  *Cluster Visualization* A scatter plot of **Annual Income vs
    Spending Score** clearly shows six groups of customers.

    -   One group has low income and low spending (Cluster 3).
    -   Another group has low income but high spending (Cluster 4).
    -   Some clusters show high-income customers with very different
        spending habits (Clusters 1, 2, and 5).
    -   The mid-income, average spenders form a stable cluster (Cluster
        0).

8.  *Insights*

    -   The model identifies both *value shoppers* (low spenders at
        all income levels) and *premium customers* (high spenders with
        high income).
    -   Businesses can target Cluster 4 and Cluster 5 for loyalty
        programs or premium product offerings.
    -   Cluster 1 customers have high income but low spending, which may
        indicate potential for marketing campaigns or personalized
        recommendations.

9.  *Output* The final dataset is saved as
    ⁠ mall_customers_with_clusters.csv ⁠, which includes each customer's
    assigned cluster for further analysis or dashboard visualization.

## How to Run the Project

1.  Place ⁠ Mall_Customers.csv ⁠ in the same folder as
    ⁠ customer_segmentation.py ⁠.

2.  Run the script:

    ⁠  bash
    python customer_segmentation.py
     ⁠

3.  The program will generate graphs and export the final CSV file.
