#!/bin/bash

aws iam create-service-linked-role --profile $USER --aws-service-name braket.amazonaws.com

aws iam create-role --profile $USER \
    --role-name AmazonBraketServiceSageMakerNotebookAccess \
    --assume-role-policy-document file://AmazonBraketServiceSageMakerNotebookAccessTrustPolicy.json

aws iam put-role-policy --profile $USER \
    --role-name AmazonBraketServiceSageMakerNotebookAccess \
    --policy-name AmazonBraketServiceSageMakerNotebookAccess \
    --policy-document file://AmazonBraketServiceSageMakerNotebookAccessPolicy.json

aws iam put-role-policy --profile $USER \
    --role-name AmazonBraketServiceSageMakerNotebookAccess \
    --policy-name AmazonBraketFullAccessPolicy \
    --policy-document file://AmazonBraketFullAccessPolicy.json
