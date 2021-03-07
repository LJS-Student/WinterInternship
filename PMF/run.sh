#!/bin/bash

for lambda_u in 0.001 0.005 0.01
do
	python Vanilla_PMF.py -lam_u ${lambda_u}
done
