# kNN-Chebyshev-Anomaly
This repo describes an experiment on investigating the anomalous phenomenon when setting the metric of kNN as Chebyshev Distance.

### Describe

When setting Chebyshev Distance
$$
dis_{che}(\mathbf{x}_i,\mathbf{x}_j)=\max_{1\leq v\leq d}(x_{i,v}-x_{j,v})
$$
as the metric of kNN, I find that the validation precision would enjoy a boost for 10%-20% when 
$$
k=\lceil|D_{train}'|/2\rceil
$$
on MNIST Dataset.

However, I could neither reproduce the experiment on any dataset with higher dimensions like Olivetti Faces,  nor on any dataset with lower dimensions like Covertype. 

MNIST is a sparse dataset and Chebyshev is a bad choice for kNN since 
$$
\begin{pmatrix}
\begin{aligned}
0\;1\;0\;0\\0\;1\;0\;0\\0\;1\;0\;0\\0\;1\;0\;0
\end{aligned}
\end{pmatrix}
$$
and 
$$
\begin{pmatrix}
\begin{aligned}
0\;1\;0\;0\\0\;1\;0\;0\\0\;1\;0\;0\\0\;1\;0\;1
\end{aligned}
\end{pmatrix}
$$
are both "1" class yet Chebyshev distance maps them to completely different spaces.

### Usage

Run the experiments on MNIST dataset to view the anomalous phenomenon that

```
python3 knnppl.py
```

The current setting of the code plots the validation precision of all 10 classes respectively. 

Run the experiments on other datasets 

```python
python3 knnface.py
```

