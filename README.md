# CS220_GNN_Accelerator

Welcome to the CS220 Research Project repository! This project aims to enhance the efficiency and scalability of Graph Neural Networks (GNNs) by optimizing memory management and computational processes for large-scale sparse graphs.

## Getting Started

To use the GNN Accelerator, follow these steps:

1. **Generate .mtx Files:**
   Run the provided Python script to generate `.mtx` files for the Cora, Citeseer and PubMed datasets.

2. **UPMEM Implementation:**
   /na. In terminal `Make`
   /nb. `./bin/spmv -f cora.mtx` (For Cora Dataset, similarly for other datasets)

4. **View Results:**
   Results will be displayed directly in the terminal.

For more details on the research and methodology behind this accelerator, please refer to our paper.

## Reference
The work presented here focuses on the optimization of matrix multiplications in GNNs using innovative memory management techniques. For further details, please see our paper.
