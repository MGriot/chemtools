# Data Types and Levels of Measurement

Understanding the different types of data is fundamental to performing correct and meaningful exploratory data analysis (EDA), statistical modeling, and visualization. This document outlines the primary data types and the four levels of measurement.

## Primary Data Types

At the highest level, data can be split into two main types:

### 1. Qualitative (Categorical) Data
This data type describes a quality or characteristic. It represents things that can be placed into distinct categories or groups.
- **Examples:** Gender (`Male`, `Female`), Country (`USA`, `Canada`), Yes/No responses.

### 2. Quantitative (Numerical) Data
This data type represents a measurable quantity, expressed as a number. Quantitative data can be further divided into:
- **Discrete Data:** Can only take on specific, distinct values (often integers). You can count this data.
  - **Examples:** Number of children in a family, number of defects in a batch.
- **Continuous Data:** Can take on any value within a given range. You can measure this data.
  - **Examples:** Height, weight, temperature.

---

## Levels of Measurement

Proposed by psychologist Stanley Smith Stevens, the four levels of measurement describe the nature of information within the values assigned to variables. Each level builds upon the previous one.

### 1. Nominal
This is the simplest level. Nominal data consists of categories that cannot be ordered or ranked. The values are just labels.

- **Key Properties:** You can count categories and determine equality (`=`, `!=`).
- **Central Tendency:** Mode.
- **Examples:**
  - Blood Type (`A`, `B`, `AB`, `O`)
  - Gender (`Male`, `Female`, `Other`)
  - Zip Code (e.g., `90210`, `10001`). Although they are numbers, you cannot perform meaningful math on them; they are just labels for a region.

### 2. Ordinal
Ordinal data consists of categories that have a meaningful order or rank, but the intervals between the ranks are not necessarily equal or known.

- **Key Properties:** You can determine order (`<`, `>`) in addition to equality.
- **Central Tendency:** Mode, Median.
- **Examples:**
  - Education Level (`High School`, `Bachelor's`, `Master's`)
  - Survey Responses (`Strongly Disagree`, `Disagree`, `Neutral`, `Agree`, `Strongly Agree`)
  - Star Ratings (`1 Star`, `2 Stars`, `3 Stars`)

### 3. Interval
Interval data is ordered, has equal intervals between values, but lacks a "true zero" point. A true zero indicates the complete absence of the attribute being measured. Because there's no true zero, ratios are not meaningful.

- **Key Properties:** You can perform addition and subtraction (`+`, `-`) in addition to ordering.
- **Central Tendency:** Mode, Median, Arithmetic Mean.
- **Examples:**
  - **Temperature in Celsius or Fahrenheit:** The difference between 10°C and 20°C is the same as between 20°C and 30°C. However, 0°C does not mean "no temperature," and 20°C is not twice as hot as 10°C.
  - **IQ Scores:** The difference between an IQ of 100 and 110 is considered the same as between 110 and 120, but there is no "zero IQ."
  - **Calendar Dates:** The interval between the year 1000 and 1500 is the same as between 1500 and 2000.

### 4. Ratio
This is the most informative level of measurement. Ratio data is ordered, has equal intervals, and possesses a true zero. This allows for meaningful ratio comparisons (e.g., "twice as much").

- **Key Properties:** You can perform multiplication and division (`*`, `/`) in addition to all previous operations.
- **Central Tendency:** Mode, Median, Arithmetic Mean, Geometric Mean.
- **Examples:**
  - **Height & Weight:** A value of 0 means no height or weight. 100 kg is twice as heavy as 50 kg.
  - **Age:** 0 is the starting point of life.
  - **Temperature in Kelvin:** 0 K is absolute zero, meaning the absence of thermal energy. 200 K is twice as hot as 100 K.
  - **Number of Children:** 0 means no children.

---

## Summary Table

| Level of Measurement | Key Properties | Examples |
| :--- | :--- | :--- |
| **Nominal** | Equality (`=`, `!=`), Categories | Blood Type, Zip Code, Gender |
| **Ordinal** | Order (`<`, `>`), Rank | Star Ratings, Education Level |
| **Interval** | Addition/Subtraction (`+`, `-`), Equal Intervals | Temperature (°C/°F), IQ Scores |
| **Ratio** | Multiplication/Division (`*`, `/`), True Zero | Height, Age, Temperature (K) |

---

## Further Reading

For a more in-depth explanation, you can refer to the Wikipedia article:
- [Level of measurement](https://en.wikipedia.org/wiki/Level_of_measurement)
