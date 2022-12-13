
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained('/BERT1', local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained('/BERT1', local_files_only=True)
text1 = r"""
Language model pre-training, such as BERT, has significantly improved the performances
of many natural language processing tasks. However, pre-trained language models are usually computationally expensive,
 so it is difficult to efficiently execute them on resource restricted devices. To accelerate inference
and reduce model size while maintaining accuracy, we first propose a novel Transformer distillation method that is 
specially designed for knowledge distillation (KD) of the Transformer-based models. 
By leveraging this new KD method, the plenty of knowledge encoded in a large teacher BERT 
can be effectively transferred to a small student TinyBERT. 
There have been many model compression techniques (Han et al., 2016) proposed to accelerate deep model inference 
and reduce model size while maintaining accuracy. The most commonly used techniques include quantization (Gong 2014),
weights pruning (Han et al., 2015), and knowl edge distillation (KD) (Romero et al., 2014). 
"""
text2=r"""
In this paper, we focus on knowledge distillation, an idea originated from Hinton et al. (2015), 
in a teacher-student framework. 
KD aims to transfer the knowledge embedded in a large teacher net work to a small student network where the student 
network is trained to reproduce the behaviors of the teacher network. Based on the framework, we propose a novel 
distillation method specifically for the Transformer-based models (Vaswani et al., 2017), and use BERT as an example 
to investigate the method for large-scale PLMs. Transformers  provide general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG)  and deep interoperability between
TensorFlow 2.0 and PyTorch. 
In this paper, we introduced a new method for Transformer-based distillation, 
and further proposed a two-stage framework for TinyBERT. Extensive experiments show that TinyBERT achieves
competitive performances meanwhile significantly reducing the model size and inference time of
BERTBASE, which provides an effective way to deploy BERT-based NLP models on edge devices.
"""


str0 = r'''
Defect management
Purpose & objectives
A defect management process is needed to guide decision-making to do the right things at the right time and to manage all defects towards resolution effectively and efficiently.
Requirements for defect management
The following are mandatory in defect management:
Document the defect management process in the master test plan.
The IT initiatives must follow the standard defect management workflow and severity classifications.
The defect management process must be communicated to and agreed with the IT initiative's key stakeholders.
The IT initiatives must use a Nordea-approved defect management tool.
All deviations from expected results as described in the test case must be recorded as defects in the defect management tool.
Standard defect management flow
This is a standard tool for independent flow, which sets minimum steps to be followed in all development initiatives. Development initiatives can add more steps in addition to the mandatory flow if they are considered useful in the project. 
Tool specific flows are defined in section on guidelines.
'''

str1 = r'''
Standard defect severity
#
Severity
Description
S1
Critical
The defect prevents   successful completion of one or more critical business processes for all or   most business users or customers. There is no workaround to achieve the expected result.
S2
Major
The defect prevents successful completion of one or more critical business processes for all or most business users or customers. However, there is workaround to achieve the expected result that can be used temporarily.
S3
Minor
The defect prevents successful completion of one or more non-critical business processes for some users, or critical business processes that are rarely used by a very limited number of business users or customers.
S4
Cosmetic
The defect does not prevent successful completion of any business process. It’s related to some inconvenience in use of system functionality or errors in the look and feel of the application or spelling mistakes.
Guides
'''

str2 = '''
Exceptional need for production data
At any time, for any testing activities, it is mandatory to use synthetic or masked data.
Usage of production data on any of  non production environments in the initiative makes the initiative non compliant with Nordea Test Strategy.
If there is a strong reason for using the production data, then following steps are required:
Step 1: Get approval from Test Data Management  team  to load  production data into any non production environment  ( by raising ticket for data consulting in ITSSP).  Test Data Management team will assess the impact of using production data in the non production environments and its connections. 
Step 2: Get authorization from the department governing data privacy (CSO)
Step 3: Get approval from the application owner (the information owner)
Step 4: Get approval by the group operational risk team and Data Protection Office (DPO).
Step 5: Create a risk record in IT Risk Tool by following the Risk Acceptance Technology Information and Security (RATIS) instructions
'''
str3 = r'''
Executive summary
Along with digital transformation, the adoption of agile ways of working and dev-ops are affecting the world of testing. Also, advanced technologies such as robotic process automation (RPA), AI and machine learning are changing how testing is performed. Technology is high on banks’ strategic agendas nowadays, and testing has therefore also become critical to business outcomes and customer satisfaction.
One of the aims of this test strategy for it to support agile ways of working and speed up the delivery chain, while at the same time keeping things simple and high-level. The strategy describes the basics of what to achieve in testing, highlighting the mandatory requirements for testing at Nordea. It is supported by a guide section – with practical instructions on how to do it. These will be continuously updated to support state-of-the-art testing. 
The Nordea test strategy 4.0 combines the Nordea software test strategy 3.0, Test environment strategy 3.0 and Test data strategy 1.1 into one. With this strategy, we want to emphasise the need to adopt greater levels of test automation to ensure fast, agile deliveries, and to conduct non-functional testing to its full extent to ensure even better product quality. 
To broaden the perspective and to provide more information about how testing fits in with other IT processes, a new section - ‘Testing by a third party’ has been added.
There are also new test metrics and KPIs defined, to be used to control and manage the quality of delivery and testing processes across development initiatives. The test documentation section now includes changes at different frequencies, different automation levels and automated reports, and the section on test tools now describes general requirements related to testing tools.
Testing is an activity that is crucial to product success and of interest to every member of the product team. Thus, this strategy has been written not only for testers and test managers, but the entire team.
'''
str4 = r'''
Functional Testing
Purpose and objectives
Functional testing is a quality assurance process for testing to determine the functionality of an IT solution.
Functionality is the capability of the IT solution to provide functions that meet stated and implied requirements when the IT solution is used under specified conditions.
Functional testing of a IT solution involves tests that evaluate functions that the system should perform. Functions are tested by feeding them input and examining the output.
Test basis can typically be described as the input material used as the basis for test analysis and design. Examples of test basis are requirements, architecture descriptions, test object design and implementation.
Typical test activities to plan for in functional testing:
Functional requirements may be described in work products such as business requirements
specifications, epics, user stories, use cases, or functional specifications, or they may be undocumented.
The functions are “what” the system should do.
Functional tests should be performed at all test levels (e.g., tests for components may be based on a
component specification), though the focus is different at each level (see section 2.2).
Functional testing considers the behavior of the software, so black-box techniques may be used to derive
test conditions and test cases for the functionality of the component or system (see section 4.2).
The thoroughness of functional testing can be measured through functional coverage. Functional
coverage is the extent to which some functionality has been exercised by tests, and is expressed as a
percentage of the type(s) of element being covered. For example, using traceability between tests and
functional requirements, the percentage of these requirements which are addressed by testing can be
calculated, potentially identifying coverage gaps.
ISTQB
Test type: A group of test activities based on specific test objectives aimed at specific characteristics of a component or system.
Regression testing = change related testing to detect whether defects have been introduced or uncovered in unchanged areas of the IT solution
Retesting = change related testing performed after fixing a defect to confirm that a failure caused by that defect does not reoccur
Smoke testing = a test suite that covers the main functionality of a component or system to determine whether it works properly before planned testing begins.
'''
str5 = '''
Vision 
As One Nordea, we want to drive a better customer experience and faster deliveries. We measure the customer experience and utilise the data for improvements. We ensure operational stability and compliance of the bank’s solutions to ensure Nordea is a trusted partner, while we drive a "built-in quality" culture and ownership.
We take best practices and technologies to build efficiencies in testing embedded in the delivery pipeline. We use automation for fast feedback enabling fact-based decision-making.
The vision is aligned with the Nordea IT strategic themes.
Scope
All IT initiatives
Different way of working – agile, SAFe, iterative, waterfall
All types of changes – software and infrastructural components, in-house developed software and purchased ones (COTS)
All kinds of IT changes (classified according to different aspects)
Applicable to all internal and external resources
Target audience
The intended audience of the Nordea test strategy is all stakeholders involved in IT deliveries
– "Everyone is responsible for testing and quality".
It is mandatory to consider all the requirements in Nordea Test Strategy. If a requirement can not be applied due to the nature of the application or system and are decided to be out of scope, it is mandatory to provide a valid justification in the IT initiative Master Test Plan.
Implement the Nordea test process by considering automation, re-usability, a risk-based approach and applying continuous improvement practices
The IT initiatives take ownership of incorporating solutions for compliance  requirements that affect the testing area. If there are any obstacles to secure compliance requirements, the IT Initiative must escalate to the next level.
If a fundamental test basis and enablers, as above, are not provided, escalation to the next level is required.
'''

str6 = '''
It is mandatory that the IT initiative, in the planning activity, identifies and organises the test process into test levels.
It is mandatory to consider all the test levels described in the Nordea test strategy when defining the test level approaches.
The applicable and non-applicable test levels are listed in the master test plan.
It is mandatory to define how end-to-end testing is performed in the context of the IT initiative.
Test levels that cannot be applied due to the characteristic of the application or system, and are decided to be out of scope, are listed with a justification.
Test level details are specified in level test plans.
 Test types (Functional and Non-functional)
It is mandatory to consider the test types stated in the Nordea test strategy when defining the test approaches per test level. 
The applicable and non-applicable test types are listed in the master test plan.
Test types that are considered as out of scope are documented with a justification.
It is mandatory to have stakeholder commitment to the selected test types.
If non-functional requirements (NFRs) are missing, escalation is mandatory.
 Quality gates and test evidence in production gate
It is mandatory to define the quality gates used and their quality criteria in the master test plan and describe how the IT initiative has implemented them. 
It is mandatory to provide test evidence to change records prior deployment to production, according to production gate requirements.
Emergency change verification process must be documented in Master Test Plan (MTP)  including sign-off procedures. 
All emergency changes must have production environment validation report (e-mail) attached to change ticket including non-production verification report when applicable.
'''
str6a = ''' 
 Defect management
The IT initiative must follow the standard defect management workflow and severity classifications.
The applied defect management workflow must be documented in the master test plan.
The defect management workflow must be communicated to and agreed with the IT initiative's key stakeholders.
All deviations from expected result as described in the test case must be recorded as defects in the defect management tool.
The IT initiatives must use a defect management tool approved to be used in Nordea.
 Test documentation
The master test plan is a document in its own right, and level test plans are either merged to MTP or are documents in their own right.
Release test plans, release test reports, test cases.
Master test reports – mandatory only for all IT development handovers between teams in a project/program closure situation or when handing over to maintenance.
 Test metrics
Implement mandatory QA execution metrics and QA capability maturity KPIs.
 Test automation
Test automation approach must be described in the master test plan.
Test automation ratio/utilisation – mandatory metric.
 Testing tools
It is mandatory to list the tools to be used in the master test plan (testing tools and tools that support testing).
It is mandatory for all IT initiatives to use tools for test management, defect management and automated execution of tests cases. Test tools should be used for:
support of the whole test process
all test levels (when applicable, if not applicable a valid reason must be given in the test documents)
all test types (functional and non-functional)
version handling of all test cases (functional and non-functional, as well as for manual and automated test cases).
 Test environments
Path from development to production via component integration, one integration and one pre-prod to ensure coherent configurations and that sufficient testing is performed.
All layers within a development, component integration, one integration or one pre-prod environment must be linked to same type of test environment (development to development, component integration to component integration, one integration to one integration, one pre-prod to one pre-prod).
 Test data
Masked production data – mandatory  to perform the steps described in the masked production data procedure.
Synthetic test data – mandatory  to perform the steps described in the synthetic data procedure.
It is mandatory to use synthetic or desensitised data at all times, for all testing activities.
It is mandatory to classify the communication with an external system to and from a Nordea test system. 
'''

str7 = '''
Approvals and business sign off
For Nordea test plans and reports it is mandatory:
to identify affected business areas and required roles for sign off
to get the approvals and sign off from identified business representatives
that the business representatives are employed in a Nordea business unit
to present the business approval and sign off evidence in the respective test document (master test plan, release test plan, release test report and master test report).
Non-functional testing
Purpose and objectives
Non-functional testing aims to minimise the risk of failure for non-functional quality attributes of the applications, products and solutions. It is the testing of a software application or system for its non-functional requirements (NFRs) – the way a system operates – rather than for the specific behavior of that system.
The objective is to substantially contribute to Nordea's customer experience excellence and operational stability.
Non-functional testing evaluates characteristics of systems and software such as usability, performance efficiency or security. Non-functional testing is the testing of “how well” the system behaves* and how the user experience the system.
Non-functional testing must be considered and executed as early as possible at all test levels, in the same way as functional testing. This means that it must be considered to be applied from the earliest testing levels and all the way through to acceptance testing.
For example, non-functional testing answers questions like:
[Performance] - How fast does this web-page load account data to the screen?
[Reliability] - How often is it not possible to log in to a system due to unplanned downtime?
[Security] - Can I trust that no unauthorized person can access my private data?
[Usability] - How easy is it to use and understand the user interface?
'''
str7a = '''
The fulfilment of the non-functional requirements (NFRs) is verified through the non-functional tests. For example, it is ensured that security controls cannot be bypassed, and that hardening of the environment has been performed.
If non-functional requirements (NFRs) are lacking, an escalation should be initiated.
At Nordea the non-functional requirements are classified by using the FURPS+ model (based on the quality attributes). The non-functional testing types are derived from the FURPS+ classification and are grouped in to the following main categories:
In the event of mandatory non-functional tests being omitted, a valid reason and justification must be provided in the master test plan (MTP).
Usability Testing
Testing to determine the extent to which the software product is understood, easy to learn, easy to operate and attractive to the users under specified conditions.
'''

str8 = '''
 Accessibility testing
Performance testing
Aims to verify the system’s performance requirements such as response time, transactional throughput and number of concurrent users support under a particular workload. It is used to measure the end-to-end performance of a system and to build the confidence in the IT solution before going live.
Performance testing is required to verify the application’s response for the intended number of users, its maximum load-resisting capacity, the application’s capacity for handling the number of transactions required for a given period by the business and the stability under expected and unexpected user load.
Performance test types listed in order of priority:
 1. Baseline testing
Serves as “snapshot” of system performance at a given acceptable load, and forms the basis of comparison with subsequent tests. 
It is expected that system performance after code change, bug fixes or for a new release, must be better or inline with baseline testing. Application or IT solution is approved to go-live based on baseline results comparison.
This test results forms the base for other release testing and is used to compare the performance after application changes with a known standard of references. 
For example, application X supports 1000 anticipated users with NFR response times, then the results become baseline. Any changes in application X should lead to better or equivalent performance compared to baseline results. In case the changes shows better performance, the new results will become baseline for next releases.
How it is performed
Same as load testing type. Load test results become baseline test results when system shows better or inline with NFR requirements.
Advantages:
It ensures consistent or better performance for every release.
It ensures system is inline with NFRs.
 2. Load testing
A type of performance testing conducted to evaluate the behavior of a component or system with expected (anticipated) load. For example, numbers of parallel users and/or total numbers of transactions per second, as well as error rate to determine that the load can be handled by the component or system.
Regression testing will be performed using this test type.
No parameters or settings are changed during regression testing. Any changes are to be reviewed and approved by IT initiative.
Approved results from load testing will become baseline/benchmark for next releases/changes.
'''
str8a = '''
How it is performed
Detailed NFRs are collected from various stakeholders. A workload model, including identified scenarios, will be designed and approved . A load test will be executed using approved scenario.
Advantages:
It ensures that the IT solution works as expected in production.
It ensures user experience is within NFR limits.
It ensures that the IT solution is in line with NFRs.
No parameters or settings are changed during regression testing. Any changes are to be reviewed and approved by IT initiative.
Approved results from load testing will become baseline/benchmark for next releases/changes.
How it is performed
Detailed NFRs are collected from various stakeholders. A workload model, including identified scenarios, will be designed and approved . A load test will be executed using approved scenario.
Advantages:
It ensures that the IT solution works as expected in production.
It ensures user experience is within NFR limits.
It ensures that the IT solution is in line with NFRs.
 3. Stress testing
A type of performance testing conducted to evaluate an IT solution/component at or beyond the limits of its anticipated or specified workloads, or with reduced availability of resources such as access to memory or servers.
How it is performed
Starting with load testing scenario, system is loaded with various unexpected load points like background jobs, increased endpoint hit rate, adding rendezvous points at various functionalities. During the stress test, other tests like fail-over, load balancer failure etc. can be performed to understand the stress condition/behavior.
Advantages:
It ensures that the IT solution can handle unpredicted loads.
It helps to understand risks and plan mitigations when unpredictable load arrives.
It provides metrics for future costs involved.
'''

str9 = '''
 4. Scalability testing
Testing to determine the scalability of the software product. Scalability is the capability of the software product to be upgraded to accommodate increased loads.
How it is performed
Starting with load testing scenario (referred as 1x), IT solution is loaded with various incremental loads (2x, 3x , 4x,.....). Thereby the IT solution is monitored for various parameters like CPU, memory, database IOPS, disk utilization, disk IO etc. A detailed analysis will be created based on the observations.
Advantages:
It ensures that the IT solution can handle future sales requirements.
It helps to understand when costing need to be reviewed.
It provides insight on any architectural changes, if required.
 5. Volume testing
Testing where the IT solution is subjected to large volumes of data. For example, database size such as million rows in a table, or processing of huge interface files (XML or JSON).
How it is performed
Similar to load testing, but pumping large volume of data into target table of database, or calling interfaces with huge size of data. The interactions can be reading and/or writing on to/from file.
Advantages:
It identifies load issues, when unpredictable data received thru interfaces.
It identifies database IO operations or table locks when processing large volume of data.
 6. Endurance testing
A type of performance testing conducted to evaluate the behavior of a component/IT solution with expected (anticipated) load for longer durations (for example 24 hrs to 1 week), to determine how the IT solution behaves in longer run, and to identify memory leaks.
How it is performed
Similar to load testing but running test execution starting from 24hrs to 1 week.
Advantages:
It ensures that there are no no bottlenecks when running longer duration.
It ensures that there are no memory leaks causing IT solution catastrophic failure.
It identifies required maintenance window for system restarts.
'''
str10 = '''
Reliability testing
Reliability defines for example the accuracy of system calculations, availability and the system's recoverability.
Reliability is the property of the application to perform without failure for a long period of time, and independent of 
external influences. Reliability tests are conducted to test the stability and consistency of the application at any given point in time.
For reliability testing, the Nordea resilience framework can be used as a guide.
The term ‘resilience’ refers to the ability of the business to adapt and respond to risks, as well as opportunities, in 
order to maintain continuous business operations, be a more attractive partner, and enable growth.
Reliability test types listed in order of priority:

'''
str10a = '''
 1. Reliability testing
The ability of the software product to perform its required functions under stated conditions for a specified period of time, or for a specified number of operations.
How it is performed
Similar to endurance test with predefined goals and duration. Goals are identified and agreed. For example, a goal can be probability of failure or length of failure.
Advantages:
To identify pattern of repeating failures.
To identify probability of failure on certain conditions.
To identify fixes for possible know failures.
 2. Failover testing
Testing by simulating failure modes or actually causing failures in a controlled environment. Following a failure, the 
failover mechanism is tested to ensure that data is not lost or corrupted, and that any agreed service levels are maintained (e.g. function availability or response times).
'''


str11 = '''
 3. Disaster/recovery testing
Testing to determine, regardless of the circumstances, the recoverability of an IT solution when a disaster occured.
How it is performed
Similar to load test with predefined load and duration, various manual controlled operations such as network disconnect, 
system shutdown and/or load balancer disconnect are performed. Aim is to study the behavior and validate that the IT 
solution performs predefined rules like auto scaling, traffic redirection etc. when interruptions occurs.
Advantages:
To identify preventive steps that reduce the risk of man-made disaster
To verify corrective measures that restore lost data and working as expected when recovery procedure performed.
To identify any potential outrage due to disaster.
 4. Recoverability testing
Testing to determine impact of system behavior after recovery. There is a slight difference compared to D/R testing. 
A recovery can be forced failure of the software in a variety of ways. Recovery testing is basically done in order 
to check how fast and better the application can recover against any type of crash or hardware failure etc.. 
Examples of such failures are restart of service after crash in predefined time, restart of pod after failure.
How it is performed
Similar to load test with predefined load and duration, various manual controlled operations such as network disconnect, 
system shutdown and/or load balancer disconnect are performed. Aim is to study the behavior and validate that the IT 
solution performs predefined rules like auto scaling, traffic redirection etc. when interruptions occurs.
Advantages:
To identify preventive steps that reduce the risk of unknow software crashes
To study end user experience during crash.
To validate browser session after crash.
'''
str11a= '''
Security testing
In accordance and in compliance with various local and international regulations for protecting customer and banking data 
(for example GDPR), security requirements are of paramount priority in the development of an IT initiative.
Applications and interfaces in scope and frequency of security testing
Security test must be performed at least every year and for all externally reachable applications and interfaces as well 
as all applications/systems that has at least two critical ratings in the Risk Impact score* for either Confidentiality,
 Integrity or Availability (CIA). If there are changes applied to the application security test must also be performed 
 before the application goes in to production unless the change is considered by Cyber Security and/or CSO to be minor.
Gateways between internal and external system must be security tested to the same extent as an Internet facing application/server.

'''

str12 = '''
Applications that have one critical Risk Impact Score for Confidentiality, Integrity or Availability must be security tested at least every three years.
For new applications (in scope for mandatory secuirty testing), the security testing must be executed, and critical as well as high classified findings (see CVSS scoring) remediated before deployment to production.
The requirements on security testing are described in Guidelines on Security testing, and Guidelines on Group Information Security Instructions (GISI), in chapter 4.10.2.3 System Testing and security review.
* available in Mega-Hopex
Security Test Certificate
All applications in scope for mandatory security testing must have a valid security test certificate issued by the Security Testing Team or a valid dispensation for deployment to production. If the application doesn't have a valid security test certificate or a valid dispensation , deployment to production will be rejected in the change management process. The security test certificate and dispensation are documented in Security Test Application (STA) tool. For more information and how to obtain and prolong the security test certificate, visit this page.
Findings (aka. vulnerabilities)
All findings detected during the security testing, must be recorded as defects in a defect management tool (e.g. JIRA) for traceability, follow up and transparency. All open defects (including security findings), must be reported in the release test report (RTR). Following is mandatory when creating defects:
Due to the sensitivity of this information details are not allowed to be written directly into the defect, therefore a reference to the security report is required! 
Major (high) or Critical findings will be automatically registered as incidents.
The time frame for fixing defects is (based on the CVSS criticality classification):
30 days for critical defects
60 days for high defects
If the identified vulnerability cannot be mitigated in the software, Application Provider must, in agreement with Application Owner, apply appropriate compensating controls to remediate risks. For more information on possible compensating controls, please read Guidelines on Security testing , chapter 4.8 Alternative mitigation.
'''
str12a = '''

If remediation is not possible within the above defined time frame or appropriate compensating controls implemented a Risk Acceptance has to be written. After the risk acceptance has been signed, it has to be reviewed and recommended/not recommended by CSE RATIS (Risk Acceptance Technology, Information and Security).
 CVSS - Common Vulnerability Scoring System
Common Vulnerability Scoring System (CVSS) is an open industry standard for assessing and scoring the severity of computer system security vulnerabilities. The CVSS scoring is mapped to five different severity classes, which are mapped to the Nordea defect severity classification as follows:
9,0 - 10	Critical	Critical
7 - 8,9	High	Major
4 - 6,9	Medium	Minor
0,1 - 3,9	Low	Cosmetic
0	None	N/A
Read more about CVSS in the Guidelines on Security testing.
How to order mandatory security testing
The mandatory security testing can be executed as external or internal security testing, and is coordinated by Security Testing Team. For more information on how the external and internal security testing is interconnected through the security certification process, and how to obtain and prolong the security test certificate, read following documents:
Process description - Security test with external supplier and/or
Process description - Security Test with internal security tester.
Maintainability (Supportability) Testing
Testing to determine the maintainability of a software product
 Portability testing
Testing to determine the portability of a software product. (Portability testing is the process of determining the degree of ease or difficulty to which a software component or application can be effectively and efficiently transferred from one hardware, software or other operational or usage environment to another.)
'''

str13 = '''
ISTQB
Non-functional testing is testing performed to evaluate that a component or system complies with non-functional requirements.
Guide
Functional and non-functional testing in Nordea Test Framework
Links
For more information on IT security at Nordea, please visit the following pages:
IT security processes
Security Portal
Information Security Documents
Guidelines on Security testing
Guidelines on Group Information Security Instructions (GISI)
For more information on security testing and how to initiate/order, please visit the following pages:
Security Test Certificates
Cyber Security Service Catalogue
How to order Penetration Test
External Security Testing (EST)
Internal Security Testing (IST)
FAQ external penetration testing
For more information on the security control gate (as part of the change process), please visit:
Change Requests for Application Changes/Deployment to Production (security approvals team)
Non-functional requirements
At Nordea, the requirements, are classified by using the FURPS+ model (based on the quality attributes):
'''

str14 = '''
Nordea Test Strategy
Welcome to Nordea's Test Strategy
Nordea Test Strategy provides all the requirements on testing at Nordea and constitutes the strategy for test. You will also find all the supportive instructive materials for testing in the strategy pages.
The test strategy is based on international best practice standards such as ISTQB and SAFe, incorporates external regulations on compliance and testing, and adheres to Nordea's IT strategy. The Nordea test strategy includes requirements on testing methodology, test environments and test data.
Introduction
Documentation & Metrics
Mandatory Items Summary
Enablers
Test Process
Nordea Test Framework
 Version history
4.0.03	December 23rd 2020	
Page Mandatory items:
Removed non-functional testing section
Test types, added following bullets
It is mandatory to consider all non-functional categories with its respective test types:
performance testing
security testing
reliability testing
maintainability testing
usability testing.
If non-functional requirements (NFRs) are missing, escalation is mandatory.
Updated structure fot Test types
Added new page Functional testing
Moved Non-functional testing under Test types
Moved content from page Test types to the two sub-pages Functional testing & Non-functional testing
Page Non-functional testing
Updated the section Security testing according to the updated requirements available in Guidelines on Security testing, and Guidelines on Group Information Security Instructions (GISI)
Clarified description for the non-functional test types performance, reliability, maintainability and usability testing.
January 26th 2021: Moved performance guide test text to Nordea Test Framework (one sentence, not a mandatory requirement).
'''
str15 = '''
4.0.02	October 21st 2020	
Layout: Improved layout for page level 1 and 2
Page Summary mandatory items:
Clarification of generic requirements regards mandatory
Page Test data management:
Clarified/simplified existing requirements
Moved procedure descriptions to NTF to simplify NTS
One additional requirement
Page Defect management:
Clarified defect work flow
Clarified/simplified existing requirements
One additional requirement
Page Quality gate and test evidence in production gate:
Two additional requirements regards testing of emergency fixes
Page Non-functional testing:
Security testing: Clarified existing requirements
4.0.01	November 5th 2019	
Updates on the following pages:
'''

str16 = '''
Test environments are a key enabler in securing a great customer experience, stable, reliable production and ongoing simplification efforts within the bank.
The purpose of this section is to provide a clear and common definition of what is required from test environments in order to have pre-conditions for effective testing in place at Nordea.
Business challenges
 Increased impact of insufficient quality
Lack of quality has a direct impact on customer experiences, and quality needs to be driven as early as possible.
Test environments must enable driving quality as early as possible.
An approach that identifies defects early in the development process needs to be promoted.
Agility for delivering faster time-to-market.
Increased competition that delivers reliable, high-quality solutions with good user experience challenges the ambition levels of the bank.
 High rate of change that balances speed and agility
Testing is key element for safeguarding a stable and well-received customer experience.
The set-up of the test environments must accommodate the required agility whilst supporting the massive amount of planned changes.
 Risk and compliance
Requirements from the ECB and FSAs require Nordea to have test environments that reflect production in terms of represented volumes, procedures and end-to-end processing.
Increased development and integration by external suppliers or third-party deliveries. More ‘open’ environments to be able to have external parties that integrate with the bank’s services (e.g. start-ups) require Nordea to be in control of the environments.
Vision
 "Test environments are the production for developers and testers and are treated in a production-like manner"
Test environments are the production for our developers and testers and should be treated in a production-like manner. Some test environments will be more production-like than others, but all will have production-like characteristics. Pre-production is the environment closest to production. Therefore, the test environments must have proper SLAs that ensure stable, secure, accessible and reliable environments. This should be done as a service with clear expectations and support from all stakeholders.
'''

str17 = '''
Nordea Test Strategy purpose and goals
Purpose and Goals
The purpose of the test strategy is to obtain a helicopter view of how to approach testing at Nordea. The objectives of the Nordea test strategy are to: 
Improve the satisfaction of customers and business.
Reduce operational risk and to increase operational stability in systems.
Ensure compliance to both external and internal regulations, policies and processes.
Support Nordea’s future journey and evolving the company maturity.

'''
str18 = '''Ensure that the IT solutions are delivered according to agreed functional and non-functional requirements.
Enable the business to take informed decisions when deciding to release new products or features.
Support fast and lean development, embracing Agile ways of working.
Connect testing with the rest of the SDLC to build high quality products at Nordea.
Ensure we have a scalable test process supporting small frequent changes and large infrequent changes.
The 9 Nordea Test Strategy objectives have been merged into 5 overall goals depicted in this picture.
'''
str19 = '''
Page Mandatory items:
Test tooling, removed following bullets:
It is mandatory to use common Nordea-approved test tools 
It is mandatory to consider the resilience requirements for test tools. 
The configurations in the test tools should support writing (e.g. writing of test results through an interface) and reading (e.g. creating reports and metrics) information in an automated way.
Added bullet Approvals and business sign off:
For Nordea test plans and reports it is mandatory:
to identify affected business areas and required roles for sign off
to get the approvals and sign off from identified business representatives
that the business representatives are employed in a Nordea business unit
to present the business approval and sign off evidence in the respective test document (master test plan, release test plan, release test report and master test report)
Renamed bullet Quality gates to Quality gates and production gate, and added one bullet:
It is mandatory to provide test evidence to change records prior deployment to production, according to production gate requirements.
Page Quality gates renamed to Quality gates and production gate
Added requirements on test evidence in production gate
Page Test Automation:
Technical changes and links to new guide
Page Test Levels:
Renamed unit integration test to component integration test
Page Test Environment Model:
Renamed unit integration test to component integration test and unit test to component test
4.0	Feb 2019	
Nordea Test strategy 4.0 combines Test strategy 3.0, Test environment strategy 3.0 and Test data strategy 1.1. A separate guide section has been created for the 'How' part.  
The main drivers for the updated version are as flows:
Alignment of the three above-mentioned strategies
New test environment target picture(approved by COO EM)
Simplification of the document
Addition of guides sections
3.0	Jun 2015	Test strategy rewritten and merged with test strategy regarding the agile development approach.
2.3	-	The sections 'Test automation' and 'Non-functional test' were updated
2.2	-	Minor updates
2.1	-	Minor updates
2.0	-	Scope extended to include non-functional testing and the use of test automation, and was updated to cover the statements in Nordea’s corporate test policy.
1.0	Dec 2008	Covered manual and functional Testing.
 Authors
Main responsible for NTS 4.0 is QA Capability and the QA PGC team, headed by Annica Johnsson
'''
str20 = '''
Pages Test Types and Test levels	
QA PGC Core team and Non-functional testing CoE, Unknown User (n493300)
Detailed page Test Types	
Non-functional testing CoE, Unknown User (n493300)
Detailed page Test Levels	
Non-functional testing CoE, Unknown User (n493300)
Summary mandatory items	
Johnsson, Annica Wegnerowska-Kulbacka, Marta Unknown User (m016639) Correia, Vinicius Pazutti Hautaniemi, Heikki Nevala, Kari
Test data management	
Johnsson, Annica Wegnerowska-Kulbacka, Marta Unknown User (m016639) Rodrigues, Spencer
Defect management	
Johnsson, Annica Correia, Vinicius Pazutti Hautaniemi, Heikki Nevala, Kari Seppänen, Pekka Ijaz, Fehmeed Boven, Roy
Quality gate and test evidence in production gate:	
Johnsson, Annica Nevala, Kari
Non-functional testing (security testing)	
Unknown User (n493300) Horvath, Sebestyen
20201021_Executive summary	
Konttila, Salla; Lena Lindman; Unknown User (n493300)
Test Process - Test Process - Agile /Traditional	
Hautaniemi, Heikki; Nevala, Kari; Johnsson, Annica
Test Process - Test Types	
Johnsson, Annica
Test Process - Test Levels	
Johnsson, Annica
Test Process - Defect Management	
Hautaniemi, Heikki; Nevala, Kari; Johnsson, Annica
Test Process - Non-functional Testing	
Pankaj Sharma
Quality Gates	
Nevala, Kari, Hautaniemi, Heikki
Test Process - Testing by Third party	
Lena Lindman
Documentation and Metrics - Test Documentation	
Poulsen, Troels, Hautaniemi, Heikki, Nevala, Kari, Unknown User (n493300), Sayadi, Mina, Johnsson, Annica
Documentation and Metrics - Test Metrics 	
Hautaniemi, Heikki
'''
str21 = '''
Nordea Test Strategy purpose and goals
Purpose and Goals
The purpose of the test strategy is to obtain a helicopter view of how to approach testing at Nordea. The objectives of the Nordea test strategy are to: 
Improve the satisfaction of customers and business.
Reduce operational risk and to increase operational stability in systems.
Ensure compliance to both external and internal regulations, policies and processes.
Support Nordea’s future journey and evolving the company maturity.
Ensure that the IT solutions are delivered according to agreed functional and non-functional requirements.
Enable the business to take informed decisions when deciding to release new products or features.
Support fast and lean development, embracing Agile ways of working.
Connect testing with the rest of the SDLC to build high quality products at Nordea.
'''

str22 = '''

Ensure we have a scalable test process supporting small frequent changes and large infrequent changes.
The 9 Nordea Test Strategy objectives have been merged into 5 overall goals depicted in this picture.
Test process - agile / traditional
Purpose and objectives
The purpose is to :
provide a clear test process that can be utilized in all IT initiatives at Nordea
create a common, scalable and efficient way of testing in all IT initiatives
build traceability of quality artefacts throughout the software development life cycle
increase test efficiency and test maturity
Nordea test process
The test process is based on the ISTQB® 'Fundamental Test process' and in alignment with the ISTQB® “Foundation Level and consists of following main activities:
Test planning (initiate)
Test planning and control
Test analysis and design
Test implementation and execution
Evaluation exit criteria and report
Close
'''
str23 = '''
To ensure a Nordea-wide standard way of planning and executing tests, it is mandatory that all IT Initiatives (agile, SAFe and traditional) follow the common test processes defined above.
 Nordea strives for transparency and fact based decisions by test documentation and metrics.
 In order to achieve this it is mandatory that all IT initiatives:
Describe how the Nordea test process is applied and implemented locally in the IT initiative and how it relates to the IT initiative software development life cycle in the master test plan (MTP)
Preserve knowledge at Nordea and enable test efficiency by documenting the high-level and low-level test approaches from which the test cases are derived, in the master test plans and level test plans
Implement the test process to enable traceability between  requirements, test cases, test case execution results, defects and configuration items
Implement the test process to enable reporting of Nordea-mandatory test metrics and KPIs
Implement the Nordea test process by considering automation, re-usability, risk-based approach and applying continuous improvement practices
The IT Initiatives take ownership of incorporating solutions for compliance requirements that affect the testing area. 
If there are any obstacles to securing compliance requirements, the IT initiative must be escalated to the next level.
If a fundamental test basis and enablers above are not provided, it must be escalated to the next level.
How to apply the test process in agile and SAFe
Built-in Quality ensures that every element and every increment of the solution reflects quality standards throughout the development life cycle.
The goal is to enable continuous testing and release on demand with fast feedback loops. The test process starts with test planning, which ensures that the master test plan is created or updated.
The detailed activities of the test process are to be executed for each requirement in the sprints, ensuring that all test types and test levels are considered in the test scope risk analysis. 
The test activities are executed in parallel with development. Release plans and reports are built in the iteration cycle on demand.
'''

str23a = '''
Continuous testing ensures baseline quality by executing regression tests as part of CI/CD pipelines.
Continuous improvements are gathered and implemented in systematic manner as part of innovation sprints and retrospectives.
How to apply the test process in a traditional set-up
The test process is applied in an IT initiative in which iterative and waterfall models are used.
The goal is to achieve control and efficiency in testing. This requires early test involvement to understand the intended IT solution, its context and risks. To enable the necessary test planning, the test process starts in the software development process initiating phase and continues in parallel along with software development activities.
ISTQB
Test Process: The set of interrelated activities comprising of test planning, test monitoring and control, test analysis, test design, test implementation, test execution, and test completion.
'''
str24 = '''
Quality gates
Quality gating is a process to define measurable checkpoints through the software development life cycle, from requirements to maintenance, and to ensure quality in deliverable.
It is mandatory to define the quality gates used and their quality criteria in the master test plan and describe how the IT initiative has implemented them. 
As a minimum, your IT initiative must implement the following as a Quality Gate:
All planned tests, including manual, automated, retest and regression tests, have passed.
No critical or major defects left unresolved
Any deviation from the above must be risk-assessed, and the risk must be accepted by the Product Owner or other Business Representative.
 Failing the above should indicate that the deliverable is not ready for production.
The last quality gate before deployment to production is the production gate, which includes the test evidence requirements. It is the business representatives responsibility to approve the provided test evidence. Change manager coordinates the risk acceptance in this quality gate according to Nordea change management process. 
IT initiatives can build upon the minimum gate presented above, as long as these are covered in the gates defined in the master test plan.
Test evidence in Production gate for changes of type "normal"
The production gate is governed by the change management process, and secures that test and risk related data are in place for all normal changes to production (in the change records). Thereby the resiliency level of an application (business criticality), together with the change impact, drives the requirements on test evidence.
The test evidence must be provided in the change record.
Requirements on test evidence for change records in production gate:
Master Test Plan (MTP) (indirectly linked through the release test plan or attached to change record)
Test evidence in production gate for changes of type "emergency"
An emergency change is ONLY used to restore a business service in production. Only change reason ‘Fix/Error’ is accepted for an emergency change. To be allowed to implement an emergency change, you need to have a reference, that you are resolving an incident as follows:
An Incident with priority 1/Critical
An Incident with priority 2/High
The following testing requirements are mandatory for critical and high incidents (emergency changes mentioned above): 
Emergency change verification process must be documented in Master Test Plan (MTP)  including sign-off procedures. 
All emergency changes must have production environment validation report (e-mail) attached to change ticket including non-prod verification report when applicable 
'''
str25 = '''
Test environments are a key enabler in securing a great customer experience, stable, reliable production and ongoing simplification efforts within the bank.
The purpose of this section is to provide a clear and common definition of what is required from test environments in order to have pre-conditions for effective testing in place at Nordea.
Business challenges
 Increased impact of insufficient quality
Lack of quality has a direct impact on customer experiences, and quality needs to be driven as early as possible.
Test environments must enable driving quality as early as possible.
An approach that identifies defects early in the development process needs to be promoted.
Agility for delivering faster time-to-market.
Increased competition that delivers reliable, high-quality solutions with good user experience challenges the ambition levels of the bank.
 High rate of change that balances speed and agility
Testing is key element for safeguarding a stable and well-received customer experience.
'''
str25a = '''
The set-up of the test environments must accommodate the required agility whilst supporting the massive amount of planned changes.
 Risk and compliance
Requirements from the ECB and FSAs require Nordea to have test environments that reflect production in terms of represented volumes, procedures and end-to-end processing.
Increased development and integration by external suppliers or third-party deliveries. 
More ‘open’ environments to be able to have external parties that integrate with the bank’s services (e.g. start-ups) require Nordea to be in control of the environments.
Vision
 "Test environments are the production for developers and testers and are treated in a production-like manner"
Test environments are the production for our developers and testers and should be treated in a production-like manner. 
Some test environments will be more production-like than others, but all will have production-like characteristics. Pre-production is the environment closest to production. 
Therefore, the test environments must have proper SLAs that ensure stable, secure, accessible and reliable environments. This should be done as a service with clear expectations and support from all stakeholders.
 "Test environments are available, and enable effective testing"
Test environments are to be available as expected and the configuration must support the requisite integration flows and test approach. 
This will enable development and maintenance teams to focus on solving business challenges in the best possible way, delivering quality in every step they take.
 "Ordering of test environments is simple and delivered as a service"
The test environments must reflect production, be up-to-date and include the test data needed. 
Test environments should be provided and supported via one clear channel with clear expectations. Test environments should, thanks to the right provisioning architecture, be easy to order and provided as a service.
 "Whatever we develop, quality is our mindset."
Even when test environments are production-like, understanding requirements and the preparation of test cases and required test data are still the keys to succeed in driving quality in the earlier stages rather than discovering issues in production.
'''
str25aa = '''

Goals
 Clear path and criteria in relation to production to deliver quality at the expected speed and with the expected agility
All applications have adapted to the test environment model as laid out in this strategy.
External developments for Nordea follow the path and criteria to reduce defects caused by those deliveries, such as offshore and commercial off-the-shelf (COTS) solutions.
All areas and initiatives have structured approaches for deployments and transition to test environments and to production.
 Sufficient ‘production-like’ test environments that enable effective testing and verification
Fully purposed test environments, in line with Nordea’s architecture and standards
Each application has a plan to secure sufficient test environments for their area of responsibility
One integrated test environment layer covering all applications, providing a ‘production-like’ setup for testing.
All business-critical applications and other core applications that form key integrations must have test environments readily available (virtualised or physical) and be provided as a stable and up-to-date service.
 '''
str25b = '''
 Available and well-managed test environments
Test environments are provided as a service with agreed service levels, requisite support and service management
All test environments have acceptable service performance according to agreed service levels
Operations management has the capacity and competence to act on service levels
Development and maintenance initiatives have a ‘test environment configuration’ that is available, up-to-date and comparable across the different test environments
 Secure and compliant test environments
For all environments it is known what is installed and who has access to the environment
For all environments, it is known what data is available and whether this complies with information security guidelines
Principles
All test environments must have relevant ‘production-like’ characteristics (closer to production, more production-like) across all components in that environment to enable effective testing.
All layers within a development, component integration, one integration or one pre-prod environment must be linked to same type of test environment (development to development, component integration to component integration, one integration to one integration, one pre-prod to one pre-prod)
All change configuration or infrastructure changes need to be tested and deployed in a controlled manner.
Test environments must be separated from the PRODUCTION environment to have a zero impact setup. This also applies for infrastructure components such as networks and firewalls.
Applications are often integrated into several business flows and are critical in both performance and functionality. Therefore, for business-critical applications and the systems on which they rely due to integration, and for systems with above-average integration, environments for both functional and non-functional testing – with a well-founded understanding of matching/proportional performance for production – must be provided.
Automated provision of test environments to be considered to reduce time and increase efficiency.
'''
# 10 is too long
str26 = '''
Robot Framework is an open source test automation framework for acceptance testing and acceptance test-driven development. 
It follows different test case styles keyword-driven, behaviour-driven and data-driven for writing test cases. 
Robot Framework provides support for external libraries, tools which are open source and can be used for automation. 
The most popular library used is Selenium Library used for web development &amp; UI testing.,
Test cases are written using keyword style in a tabular format.
 You can use any text editor or Robot Integrated Development Environment (RIDE) for writing test cases., 
 Robot framework works fine on all the Operating Systems available. 
 The framework is built on Python and runs on Jython (JVM) and IronPython (.NET)., 
 In this section, we will look at the different features offered by Robot.
 Robot framework comes with a simple tabular format where the test cases are written using keywords. 
 It is easy for a new developer to understand and write test cases., 
 Robot framework comes with built-in keywords available with robot framework, keywords available from the libraries like
Selenium Library (open browser, close browser, maximize browser, etc.). We can also create user-defined keywords, which are
 a combination of other user-defined keywords or built-in or library keywords. We can also pass arguments to those keywords, 
 which make the user-defined keywords like functions that can be reused., Robot framework supports variables – scalar, 
 list and dict. Variables in robot framework are easy to use and are of great help while writing complex test cases., 
 Robot framework has support for a lot of external libraries like SeleniumLibrary, Database Library, FTP Library and http library. 
 SeleniumLibrary is mostly used as it helps to interact with the browsers and helps with web application and UI testing. 
 Robot framework also has its own built-in libraries for strings, date, numbers etc., 
 '''
str27 = '''
 Robot framework also allows the import of robot files with keywords externally to be used with test cases. 
 Resources are very easy to use and are of great help when we need to use some keywords already written for other test projects.
 , Robot framework supports keyword driven style test cases and data driven style. 
 Data driven works with high-level keyword used as a template to the test suite and the test cases are used to share data with 
 the high-level keyword defined in the template. It makes the work very easy for testing UI with different inputs., 
 Robot framework allows to tag test-cases so that we can either run the tags test-cases or skip the tagged testcases. 
 Tagging helps when we want to run only a group of test cases or skip them.,
  Robot framework provides all the details of test suite, test case execution in the form of report and logs. 
  All the execution details of the test case are available in the log file.
  '''
str28 = '''
The details like whether the test case has failed or passed, time taken for execution, steps followed to run the test 
   case are provided., This editor available with Robot framework helps in writing and running test cases. 
   The editor is very easy to install and use. RIDE makes life easy for writing test cases by providing framework specific 
   code completion, syntax highlighting, etc. Creation of project, test suite, test case, keywords, variables, importing library,
    executing, tagging the test case is easily done in the editor. Robot framework also provides plugins for eclipse, 
    sublime, Textmate, Pycharm that has support for robot test cases., Robot framework is open source, so anyone who wants 
    to try out can easily do so., It is very easy to install and helps in creating and executing test cases.
     Any new comer can easily understand and does not need any high level knowledge of testing to get started with robot framework.
     , It supports keyword-driven, behaviour-driven and data-driven style of writing test cases.
      It is a good support for external libraries. Most used is Selenium Library, which is easy to install and use in robot framework.
       Robot lacks support for if-else, nested loops, which are required when the code gets complex.
       Robot Framework is an open source test automation framework for acceptance testing and acceptance test-driven development. 
    The test cases in Robot Framework are based on keywords written in tabular format, which makes it clear and readable, 
    and conveys the right information about the intention of the test case. For example, to open browser, 
    the keyword used is Open Browser[Robot framework is built using python. In this chapter, we will learn how to set up Robot 
    Framework. To work with Robot Framework, we need to install the following To install python,
     go to python official site and download the latest version or the prior version of python as per your operating system 
     (Windows, Linux/Unix, Mac, and OS X) you are going to use.
'''
str29 = '''  
 Here is the screenshot of the python download site, The latest version available as per release dates are as follows , 
 Before you download python, it is recommended you check your system if python is already present by running the following command 
 in the command line −, If we get the version of python as output then, we have python installed in our system. 
 Otherwise, you will get a display as shown above., Here, we will download python version 2.7 as it is compatible to the windows 8
  we are using right now. Once downloaded, install python on your system by double-clicking on .exe python download. 
  Follow the installation steps to install Python on your system. Once installed, to make python available globally, 
  we need to add the path to environment variables in windows as follows −, Right-click on My Computer icon and select properties.
   Click on Advanced System setting and the following screen will be displayed., 
   Click on Environment Variables button highlighted above and it will show you the screen as follows −, 
   Select the Variable Path and click the Edit button., Get the path where python is installed and add 
   the same to Variable value at the end as shown above., 
   Once this is done, you can check if python is installed from any path or directory as shown below 
   '''
str30 = '''
is the path of the Python directory, Now, we will check for the next step, which is pip installation for python. 
PIP is a package manager to install modules for python., PIP gets installed along with python and you can check the same in 
command line as follows −, Here we are still not getting the version for pip. We need to add the pip path to 
Environment variables so that we can use it globally. PIP will be installed in Scripts folder of python as shown below −
, Go back to environment variables and add the path of pip to the variables list. 
Add C:\Python27\SCripts to environment variables as follows −, Now open your command line and check the version of pip installed −, 
So now, we have python and pip installed., We will now use pip – python package manager to install the robot framework 
and the command for it is as follows −, Once the installation is done, you can check the version of robot framework installed as shown
 below −, So, we can see Robot Framework 3.0.4 is installed., We need wxPython for Robot Framework Ride, which is an IDE for 
 Robot Framework., For windows to get the required download for wxPython, go to the following URL −, 
 And, download 32 or 64-bit wxpython for windows as per your Windows Operating system., 
 Download the 32-bit wxPython and install the same., Once the installation is done, it opens the command line and auto runs some commands as shown below −, wxPython is now installed.This module is required for the RIDE Ide to be used for Robot Framework which is the next step., <b>On Linux</b>, you should be able to install wxPython with your package manager. For
example, on Debian based systems such as Ubuntu running sudo apt-get install pythonwxgtk2.8 ought to be enough.
'''
str31 = '''On OS X, you should use wxPython binaries found from the wxPython download page. wxPython2.8 only has 
32 bit build available, so Python must be run in 32-bit mode also. This can be done globally by running, 
On OS X, or, just for the RIDE execution −, Ride is Robot Framework IDE. We can use pip to install it as shown below.
Once the installation is done, open the command prompt and type the following command to open the Ride-IDE.
The above command opens the IDE as follows −, So we are done with the installation of Robot Framework and can get started 
working with it. We now know how to install python, pip, robot framework and also get RIDE installed to work with test cases 
in robot framework.
'''
str32 = '''
Ride is a testing editor for Robot Framework. Further, we will write test cases in Ride. To start Ride, 
we need to run the command shown below., The above command will open the IDE as shown in the following screenshot
 −, In this chapter, we will walk through the editor to see what options and features are available in the IDE. 
 The options and features will help us in testing our project., Go to File and click on New Project as shown below −, 
 The following screen will appear when you click New Project., Enter the name of the project. 
 Created Path is the path where the project will get saved. You can change the location if required.
  The project can be saved as File or directory. You can also save the project in format like ROBOT, TXT, TSV or HTML. 
  In this tutorial, we are going to use the format ROBOT and how to write and execute test-cases.,
   Now, we will add a project as a file the way it is shown below. The project is named Testing and the following screen 
   appears after the project is created., The name of the project is shown on the left side and on the right side we can see 
   three tabs Edit, TextEdit and Run., Edit has a lot of options on the UI as shown above. 
   In this section, we can add data required to run our test cases. We can import Library, Resource, Variables, 
   Add scalar, Add list, Add dict and Add Metadata., 
   The details added in the Edit section will be seen in the next tab, Text Edit.
 '''
str33 = '''
    You can write the code here in text edit section., If there is any change added in Textedit, it will be seen 
    in the Edit section. Therefore, both the tabs Edit and TextEdit are dependent on each other and the changes done will 
    be seen on both. Once the test cases are ready, we can we use the third tab Run to execute them., 
    The Run UI is as shown above. It allows to run the test case and comes with options like start, stop, pause continue, 
    next test case, step over, etc. You can also create Report, Log for the test cases you are executing., 
    To create a test case, we have to do the following −, Right-click on the project created and click on new test case 
    as shown below −, Upon clicking New Test Case, a screen appears as shown below −, Enter the name of the test case and 
    click OK. We have saved the test case as TC0. The following screen appears once the test case is saved., 
    The test case has options like Documentation, setup, teardown, tags, timeout and Template. They have an edit button 
    across it; upon clicking the button a screen appears wherein, you can enter the details for each option. 
    We will discuss the various parameters of these details in our subsequent chapters.
    The test cases can be written in tabular format as shown below. 
    Robot framework test cases are keyword based and we can write the test-cases using built-in keywords or
     keywords imported from the library. We can also create user-defined keywords, variables, etc. in robot framework.,
      There are shortcuts available in the navigation bar to run/stop test case as shown below −,
       The search keyword option can be used as shown in the screenshot below −
'''
str34 = '''
To get the list of keywords available with robot framework, simple press ctrl+space in the tabular format as shown 
below and it will display all the keywords available −, In case, you cannot remember the keyword, 
this will help you get the details. We have the details available across each keyword. 
The details also show how to use the related keyword. In our next chapter, we will learn how to create our first 
test case in ride., In this chapter, we have seen the features available with RIDE. We also learnt how to create 
       test cases and execute them.
We will explore RIDE and work on our first test case., Open Ride from command prompt or you can create a shortcut of 
ride on your desktop., Go to the path where ride is installed; for windows, it is C:\Python27\Scripts.,
 Right-click on ride.py and click Send To -&gt; Desktop (create shortcut)., 
 You will now see an icon of ride on your desktop. You can click on it to open the ride editor., 
 Let us start with our first test case in ride. Open the editor and click on File -&gt; New Project., 
 Click on New Project and enter the name of the project., 
 Parent Directory is the path where the project will be saved. You can change the path if required. 
 I have created a folder called robotframework and will save all the files in that folder.
'''
str35 = '''

 Project FirstTestCase is created., To create test case, right-click on the project., Click New Test Case.,
  Enter the name of the test case and click OK., There are 3 tabs shown for the test case created − Edit, Text Edit and 
  Run., The Edit tab comes with two formats – Settings and Tabular. We will discuss the two formats in our subsequent sections.
  In Settings, we have documentation, setup, teardown, tags, timeout and template. 
  You can add details about your test case so that it becomes easy for future reference., Click OK to save the documentation., 
  If there is a setup assigned to a test case, it will be executed before the test case execution and the test setup that 
  will be executed after the test case is done for teardown. We will get into the details of this in our subsequent chapters. 
  We do not need it now for our first test case and can keep it empty., This is used for tagging test cases – to include, 
  exclude specific test cases. You can also specify if any of the test cases is critical., This is used to set a timeout on the 
  test case. We will keep it empty for now., This will have the keywords to be used for the test case. 
  It is mostly used for data driven test case. The high-level user-defined keyword is specified in the template 
  and test cases are used to pass data to the keyword., In the tabular format, we will write our first test case and execute the same 
  to see the output., In this test case, we are just going to add some logs and see the output of it. 
  Consider the following screenshot to understand this −, We have used the keyword Log to log messages as shown above., 
  Based on the keywords specified in Edit, we can get the code in Text Edit as shown below −, 
  You may also write the test case in the Text Edit and the same will reflect in the tabular format. 
 '''
str36 = '''
  Now let us Run the test case and see the output., To run the test case, we need to click on Start as shown below −, 
  Click on start and here are is the output of the test case −, Our test case has executed successfully and the details 
  are as shown above. It gives the status as PASS., We can also see the details of the test case execution in
   Report and Log as highlighted below., Click on Report and it opens the details in a new tab as follows, In Report, 
   it gives the details like the start time, end time, path to the log file, status of the test case, etc., Click on 
   Log at the top right corner in report or from the Run screen., Here are the details of the log file −, 
   The Log file gives the details of the test execution and the details of keywords we gave for the test case., 
   In the report and the log file, we get green color for the status., Let us now make some changes that will lead to 
   the failure of the test case fail and see the output., In the above test case, the Log keyword is wrong. 
   We will run the test case and see the output −, We see that the test case has failed. I have highlighted the error 
   that it tells about the test case., Now will see the report and log output.
'''
str37 = '''
When the test case fails, the color is changed to Red as shown above.
In this chapter, we covered a simple test case and the results seen during execution are shown. 
The reports and logs show the details of test case execution.
In this chapter, we will learn how to write and execute test cases. We would cover the following areas in this chapter −,
 Run the command ride.py to start RIDE IDE., Click on File; New Project as shown below −, Upon clicking New Project, 
 the screen will appear as shown below −, New Project shows the type as file or directory. By default, File is selected. 
 We will click on Directory to create test suite, which can have many test suites in that directory. 
 Each suite will have test−cases., We will use the ROBOT format for now., The Parent-Directory is the path where the 
 WritingTestCases directory will be created. Click OK to save the test suite directory.
  Right-click on the directory created and click on New Suite. You can also create sub directories with test suites in that.
   For now, we will start with Test Suite creation as shown below −, , Click OK to save the Test suite., 
   Now you can add test case to the suite. Right-click on the Test suite created as shown below −, 
   Click New Test Case. It will display the screen to add name of the test case as shown below −, 
   Click OK to save the test case. We have the project setup ready., Robot Framework has its own built-in library, 
   which need not be imported. But we need to interact with the browsers, databases, etc. 
   To interact, we need to import the libraries.
'''

texts=[text1,text2, str0, str1, str2, str3, str4, str5, str6,str6a, str7, str7a, str8, str8a, str9, str10,str10a, str11,str11a, str12, str12a,
             str13, str14, str15, str16,
             str17, str18, str19, str20, str21, str22, str23, str23a, str24, str25, str25a, str25aa, str25b, str26, str27, str28, str29,
             str30, str31, str32, str33, str34, str35, str36, str37]





questions = [
    " What is BERT ?",
    " What is purpose of Transformer distillation ?",
    " What is goal of Transformer distillation ?",
    " What is aim of Transformer distillation ?",
    " What does Transformer distillation do ?",
    " Is is difficult to execute pre-trained language models ?",
    " What happens with knowledge encoded in a large teacher BERT ?",
    " Give examples of  Transformer architectures ?",
    "What does Transformers provide?",
    "what is quantization ?",
    "what is Vaswani known for ?",
    "what did Vaswani do ?",

    " who is Vaswani ?",

    "what is Hinton known for ?",
    "what did Hinton do ?",
    " whose idea was teacher-student framework ?",
    " who is Hinton ?",
    " How many compression techniques exist ?",

    "Transformers provides interoperability between which frameworks?",
    
 " What is  Robot Framework ?",
 " what is Selenium Library ?",
 "what text editor can you use in Robot framework ?",
  " What shows the details of test case execution ? "
 
 
    
]
i=0
for question in questions:

    for text in texts:
        inputs = tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        pred = model(**inputs)
        answer_start_scores, answer_end_scores = pred['start_logits'][0], pred['end_logits'][0]

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        print(f"Question: {question}")
        print(f"Answer: {answer}")

        print(f" text: {i}")
#        print(f" text: {varname(text)}")
    i=i+1