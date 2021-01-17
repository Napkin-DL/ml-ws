# ml-ws

### CloudFormation을 활용한 AWS Resource 생성

이 워크샵에 필요한 AWS 리소스를 생성하기 위해 CloudFormation 스택을 제공합니다.\
CloudFormation 스택은 실습환경으로 사용할 SageMaker 노트북 인스턴스를 생성하게 되며, 실습에 활용할 서비스을 실행하는 policy가 포함된 Role을 함께 생성하게 됩니다.\
아래 링크를 선택하면 스택이 시작될 AWS 콘솔의 CloudFormation 으로 자동 redirection됩니다.

* [Launch CloudFormation stack (N.Virginia)](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/create/review?stackName=AIMLWorkshop&templateURL=https://napkin-share.s3.ap-northeast-2.amazonaws.com/cloudformation/sagemaker-hol.yml)


CloudFormation 스택은 최소한 다음 리소스를 만듭니다.

* Jupyter 노트북에서 모델을 정의하는 SageMaker 노트북 인스턴스. 모델 자체는 AWS SageMaker 서비스를 사용하여 학습됩니다.
* AWS 리소스에 액세스하는 데 필요한 IAM 역할

AWS CloudFormation 콘솔의 Quick create stack 페이지로 리디렉션 된 후 다음 단계를 수행하여 스택을 시작하십시오.

* Intial 에서 자신의 영문 initial 또는 unique한 영문글자를 넣어주시기 바랍니다.
* Capabilities 에서 ***I acknowledge that AWS CloudFormation might create IAM resources***을을 체크합니다.
* Create stack 버튼을 누르고, stack 생성이 완료될 때까지 기다립니다. 10분 정도 소요됩니다.

![fig1.png](images/fig1.png)

CloudFormation 콘솔의 스택에 대한 ***Output*** 섹션에서 생성 된 리소스에 대한 정보를 찾을 수 있습니다. 언제든지 ***Output*** 섹션으로 돌아와서 값을 확인할 수 있습니다.

![fig2.png](images/fig2.png)

***Output*** 섹션에서 링크를 클릭하시면 SageMaker Notebook으로 접근이 가능하시며, 실습한 코드를 사전에 준비하였습니다.

지금까지 환경설정이 완료되었고 다음 장부터 Modeling과 Inference를 위한 실습이 진행됩니다.
