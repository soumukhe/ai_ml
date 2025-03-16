# Setting Up a Kubernetes Cluster on AWS using eksctl

## Prerequisites
- An AWS account
- An **EC2 instance (Jumpbox)**
- AWS CLI installed and configured
- **kubectl** and **eksctl** installed on the EC2 instance

---

## Step 1: Set Up EC2 Instance (Jumpbox)
Launch an EC2 instance and SSH into it.

```sh
sudo apt update -y
```

---

## Step 2: Install AWS CLI v2
```sh
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

### Configure AWS CLI
```sh
aws configure
```
Enter the following details:
- **Access Key**: `your-access-key`
- **Secret Key**: `your-secret-key`
- **Default region**: Same as EC2 instance (e.g., `us-west-2`)
- **Output format**: `json`

---

## Step 3: Install `kubectl`
```sh
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
```

---

## Step 4: Install `eksctl`
```sh
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin
```

---

## Step 5: Create an EKS Cluster

### Create `cluster.yaml`
```sh
cat > cluster.yaml << 'EOF'
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: slurm-demo
  region: us-west-2  # Change this to your preferred region

nodeGroups:
  - name: ng-1
    instanceType: t3.medium
    desiredCapacity: 2
    minSize: 2
    maxSize: 3
    labels:
      role: worker
    iam:
      withAddonPolicies:
        imageBuilder: true
        autoScaler: true
        ebs: true
        albIngress: true

# Enable control plane to serve as worker node
managedNodeGroups:
  - name: control-workers
    instanceType: t3.medium
    desiredCapacity: 1
    minSize: 1
    maxSize: 1
    labels:
      role: control-worker
    taints: []  # No taints to allow pods to schedule here
EOF
```

### Create the Cluster
```sh
eksctl create cluster -f cluster.yaml
```

---

## Step 6: Managing the Cluster

### **Delete the EKS Cluster**
(*Do not run this now unless you intend to delete the cluster*)
```sh
eksctl delete cluster --name slurm-demo --region=us-west-2
```

### **Scaling Node Groups**

#### **Change Minimum Size of Worker Nodes**
```sh
eksctl scale nodegroup --cluster=slurm-demo --name=ng-1 --nodes-min=0 --nodes-max=3 --region=us-west-2
```

#### **Scale Down Worker Nodes to 0**
```sh
eksctl scale nodegroup --cluster=slurm-demo --name=ng-1 --nodes=0 --region=us-west-2
```

#### **Change Minimum Size of Master Nodes**
```sh
eksctl scale nodegroup --cluster=slurm-demo --name=control-workers --nodes-min=0 --nodes-max=1 --region=us-west-2
```

#### **Scale Down Master Nodes to 0**
```sh
eksctl scale nodegroup --cluster=slurm-demo --name=control-workers --nodes=0 --region=us-west-2
```

### **Check Node Group Status**
```sh
eksctl get nodegroup --cluster slurm-demo --region us-west-2 --name ng-1
eksctl get nodegroup --cluster slurm-demo --region us-west-2 --name control-workers
```

### **Scale Up Nodes Again**
```sh
eksctl scale nodegroup --cluster=slurm-demo --name=ng-1 --nodes=2 --region=us-west-2
eksctl scale nodegroup --cluster=slurm-demo --name=control-workers --nodes=1 --region=us-west-2
```

---

## **Summary**
This guide walks through:
- Setting up an **EC2 Jumpbox**
- Installing **AWS CLI, kubectl, eksctl**
- Creating an **EKS cluster** using `eksctl`
- Managing the cluster by scaling nodes up/down
- Checking and deleting the cluster
