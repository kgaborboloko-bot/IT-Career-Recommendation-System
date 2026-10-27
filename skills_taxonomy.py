# skills_taxonomy.py

import re
import unicodedata
from difflib import get_close_matches

TRACKS = [
    "Business Analysis & Delivery",
    "Cloud & DevOps",
    "Cybersecurity",
    "Data & Analytics",
    "ERP/Enterprise",
    "Software Development",
    "Support & Infrastructure",
    "Testing & QA"
]

TRACK_ALIASES = {
    "Security": "Cybersecurity",
    "DevOps & Cloud": "Cloud & DevOps",
    "Testing": "Testing & QA"
}


def canonical_track(name):
    name = str(name).strip()
    return TRACK_ALIASES.get(name, name)


# Each category contains canonical skill names.
# A skill can also be associated with other specialisations.

SKILL_GROUPS = {

    "Software Development": [
        "Python", "Java", "JavaScript", "TypeScript", "C#",
        "C++", "C", ".NET", "PHP", "Ruby", "Go", "Rust",
        "Kotlin", "Swift", "Scala", "R", "Bash",
        "PowerShell", "HTML", "CSS", "React", "Angular",
        "Vue.js", "Node.js", "Next.js", "Express.js",
        "Django", "Flask", "FastAPI", "Laravel",
        "Spring Boot", "ASP.NET", "REST APIs",
        "GraphQL", "gRPC", "Microservices", "Git",
        "GitHub", "GitLab", "Bitbucket", "Maven",
        "Gradle", "Object Oriented Programming",
        "Design Patterns", "Software Architecture",
        "API Integration", "Code Review",
        "Android", "iOS", "Flutter", "React Native",
        "Xamarin", "Unity", "Unreal Engine",
        "Embedded Systems", "MATLAB", "Delphi",
        "Groovy", "Perl", "COBOL", "VBA",
        "Bootstrap", "jQuery", "Blazor",
        "Entity Framework", "LINQ", "MVC",
        "SOAP", "Web Development", "Mobile Development"
    ],

    "Data & Analytics": [
        "SQL", "Python", "R", "Pandas", "NumPy",
        "Scikit-learn", "TensorFlow", "PyTorch",
        "Keras", "Machine Learning", "Deep Learning",
        "NLP", "Computer Vision", "Generative AI",
        "Power BI", "Tableau", "QlikView",
        "Qlik Sense", "Looker", "Excel",
        "Power Query", "DAX", "Data Analysis",
        "Data Modelling", "Data Visualisation",
        "Statistical Analysis", "Statistical Modelling",
        "Feature Engineering", "Data Cleaning",
        "Data Engineering", "Data Warehousing",
        "Data Governance", "Data Quality",
        "Data Lineage", "Big Data", "Spark",
        "Hadoop", "Kafka", "Airflow",
        "Databricks", "Snowflake", "ETL",
        "SSIS", "SSRS", "SSAS", "dbt",
        "Delta Lake", "MLflow", "MLOps",
        "Model Monitoring", "Hugging Face",
        "LangChain", "Vector Databases",
        "Retrieval Augmented Generation",
        "Prompt Engineering", "Time Series Analysis",
        "A/B Testing", "Data Mining",
        "Predictive Analytics", "Business Intelligence",
        "Jupyter Notebook", "MongoDB", "PostgreSQL", "MySQL",
        "Microsoft SQL Server", "Google Colab", "Oracle Database"
    ],

    "Cybersecurity": [
        "Network Security", "Cybersecurity",
        "Information Security", "Ethical Hacking",
        "Penetration Testing", "Nmap", "Wireshark",
        "Burp Suite", "Metasploit", "Kali Linux",
        "OWASP Top 10", "Vulnerability Assessment",
        "Vulnerability Management",
        "Incident Response", "Threat Hunting",
        "Threat Intelligence", "Digital Forensics",
        "SIEM", "Splunk", "Microsoft Sentinel",
        "Identity and Access Management",
        "Multi-Factor Authentication",
        "Cryptography", "Encryption",
        "Security Operations Center",
        "Security Auditing", "Zero Trust",
        "NIST Cybersecurity Framework",
        "Cloud Security", "Application Security",
        "Endpoint Security", "Firewall",
        "Intrusion Detection", "Intrusion Prevention",
        "Security Monitoring", "Risk Assessment",
        "Security Compliance", "ISO 27001",
        "Security Awareness", "Access Control",
        "Packet Analysis", "Malware Analysis"
    ],

    "Cloud & DevOps": [
        "AWS", "Azure", "Google Cloud",
        "Cloud Computing", "Cloud Architecture",
        "Docker", "Docker Compose", "Kubernetes",
        "Terraform", "Ansible", "Jenkins",
        "CI/CD", "DevOps", "GitHub Actions",
        "GitLab CI", "Azure DevOps",
        "Infrastructure as Code", "Serverless",
        "AWS Lambda", "Azure Functions",
        "CloudFormation", "Pulumi", "Argo CD",
        "Prometheus", "Grafana", "Datadog",
        "OpenTelemetry", "Nginx", "Linux",
        "Linux Administration", "Bash",
        "PowerShell", "PowerShell Scripting",
        "Cloud Networking", "Cloud Security",
        "Containerisation", "Load Balancing",
        "Monitoring", "Site Reliability Engineering",
        "Microservices", "Helm"
    ],

    "Support & Infrastructure": [
        "Cisco Packet Tracer", "GNS3", "EVE-NG",
        "Cisco IOS", "Cisco CCNA", "Cisco CCNP",
        "Networking", "Network Administration",
        "Network Architecture", "TCP/IP",
        "DNS", "DHCP", "VLAN", "Subnetting",
        "Routing", "Switching", "OSPF", "BGP",
        "SD-WAN", "VoIP", "Wireshark",
        "Network Monitoring", "Network Troubleshooting",
        "Network Simulation", "Network Security",
        "Linux", "Windows Server", "Active Directory",
        "Microsoft Entra ID", "Microsoft 365",
        "Intune", "Exchange", "VMware",
        "Hyper-V", "Virtualization",
        "System Administration", "Technical Support",
        "Service Desk", "ITIL", "Troubleshooting",
        "Hardware Support", "Software Installation",
        "Endpoint Management", "Backup and Recovery",
        "Disaster Recovery", "Storage Area Networks",
        "Load Balancing", "Remote Desktop",
        "Operating Systems"
    ],

    "Business Analysis & Delivery": [
        "Business Analysis", "Requirements Gathering",
        "Requirements Engineering", "Business Process Modelling",
        "Process Mapping", "Process Modelling",
        "BPMN", "UML", "Agile", "Scrum",
        "Kanban", "Jira", "Confluence",
        "User Stories", "Stakeholder Management",
        "Change Management", "Project Management",
        "Product Management", "Business Intelligence",
        "Data Visualisation", "SQL", "Excel",
        "Power BI", "Figma", "UI/UX Design",
        "Wireframing", "Prototyping",
        "User Acceptance Testing", "Documentation",
        "Process Improvement", "Systems Analysis",
        "Business Requirements", "Functional Requirements"
    ],

    "ERP/Enterprise": [
        "SAP", "SAP ABAP", "SAP HANA",
        "SAP Basis", "SAP Fiori", "SAP S/4HANA",
        "Salesforce", "Salesforce Apex",
        "ServiceNow", "Dynamics 365",
        "Microsoft Dynamics 365", "Oracle ERP",
        "Oracle Fusion", "JD Edwards",
        "MuleSoft", "Dell Boomi", "Workday",
        "Power Platform", "SharePoint",
        "Enterprise Integration", "ERP Configuration",
        "ERP Implementation", "CRM",
        "Enterprise Architecture"
    ],

    "Testing & QA": [
        "Software Testing", "Manual Testing",
        "Test Automation", "Selenium",
        "Cypress", "Playwright", "JUnit",
        "Pytest", "Postman", "API Testing",
        "Integration Testing", "Unit Testing",
        "Regression Testing", "Performance Testing",
        "Load Testing", "Accessibility Testing",
        "User Acceptance Testing", "Test Planning",
        "Test Cases", "Bug Tracking",
        "Quality Assurance", "Test Driven Development",
        "Behaviour Driven Development",
        "JMeter", "Appium", "A/B Testing"
    ]
}


# Alternative names and spellings.
# Add more aliases here without changing the main skill names.

SKILL_ALIASES = {

    "Cisco Packet Tracer": [
        "packet tracer",
        "cisco packet tracer"
    ],

    "Data Modelling": [
        "data modeling",
        "data modelling"
    ],

    "Data Visualisation": [
        "data visualization",
        "data visualisation",
        "data viz"
    ],

    "Machine Learning": [
        "machine learning",
        "machine-learning"
    ],

    "Power BI": [
        "powerbi",
        "power bi desktop"
    ],

    "Scikit-learn": [
        "sklearn",
        "scikit learn"
    ],

    "JavaScript": [
        "java script",
        "js"
    ],

    "TypeScript": [
        "typescript",
        "ts programming"
    ],

    "C#": [
        "c sharp",
        "csharp"
    ],

    "C++": [
        "cpp",
        "c plus plus"
    ],

    ".NET": [
        "dotnet",
        "asp.net"
    ],

    "Node.js": [
        "nodejs",
        "node js"
    ],

    "Vue.js": [
        "vue",
        "vuejs"
    ],

    "React": [
        "reactjs",
        "react.js"
    ],

    "REST APIs": [
        "rest api",
        "restful api",
        "rest apis"
    ],

    "SQL": [
        "structured query language",
        "t-sql",
        "pl/sql"
    ],

    "PostgreSQL": [
        "postgres"
    ],

    "AWS": [
        "amazon web services"
    ],

    "Azure": [
        "microsoft azure"
    ],

    "Google Cloud": [
        "gcp",
        "google cloud platform"
    ],

    "Kubernetes": [
        "k8s"
    ],

    "CI/CD": [
        "cicd",
        "ci cd",
        "continuous integration and continuous delivery"
    ],

    "Infrastructure as Code": [
        "iac"
    ],

    "Identity and Access Management": [
        "iam"
    ],

    "Multi-Factor Authentication": [
        "mfa",
        "2fa"
    ],

    "SIEM": [
        "security information and event management"
    ],

    "NLP": [
        "natural language processing"
    ],

    "ETL": [
        "extract transform load"
    ],

    "MLOps": [
        "ml ops",
        "machine learning operations"
    ],

    "Retrieval Augmented Generation": [
        "rag"
    ],

    "Object Oriented Programming": [
        "oop",
        "object-oriented programming"
    ],

    "User Acceptance Testing": [
        "uat"
    ],

    "Quality Assurance": [
        "qa"
    ],

    "Business Intelligence": [
        "bi"
    ],

    "Excel": [
        "microsoft excel",
        "ms excel"
    ],

    "Jira": [
        "jira software"
    ],

    "VLAN": [
        "vlans",
        "virtual lan"
    ],

    "TCP/IP": [
        "tcp ip",
        "tcpip"
    ],

    "Linux Administration": [
        "linux admin"
    ],

    "PowerShell Scripting": [
        "powershell scripts"
    ]
}


def key(value):

    value = unicodedata.normalize(
        "NFKC",
        str(value)
    )

    value = value.casefold().strip()

    value = value.replace("–", "-")
    value = value.replace("—", "-")

    return " ".join(value.split())


# Build a unique list of canonical skills.

ALL_SKILLS = sorted(
    set(
        skill
        for skills in SKILL_GROUPS.values()
        for skill in skills
    )
)


# Build a canonical dictionary.

TECH_SKILLS = {
    skill: SKILL_ALIASES.get(skill, [])
    for skill in ALL_SKILLS
}


# Build a mapping from each skill to all its specialisations.

SKILL_SPECIALISATIONS = {}

for track, skills in SKILL_GROUPS.items():

    for skill in skills:

        if skill not in SKILL_SPECIALISATIONS:
            SKILL_SPECIALISATIONS[skill] = {}

        SKILL_SPECIALISATIONS[skill][track] = 1.0


# Example: networking tools may also be relevant to cybersecurity.

SKILL_SPECIALISATIONS["Cisco Packet Tracer"][
    "Cybersecurity"
] = 0.45

SKILL_SPECIALISATIONS["Wireshark"][
    "Cybersecurity"
] = 1.0

SKILL_SPECIALISATIONS["Linux"][
    "Cybersecurity"
] = 0.6

SKILL_SPECIALISATIONS["Python"][
    "Cybersecurity"
] = 0.5

SKILL_SPECIALISATIONS["Python"][
    "Data & Analytics"
] = 1.0

SKILL_SPECIALISATIONS["SQL"][
    "Software Development"
] = 0.8

SKILL_SPECIALISATIONS["Docker"][
    "Software Development"
] = 0.6


# Build alias lookup and identify ambiguous aliases.

ALIAS_TO_SKILL = {}

ALIAS_COLLISIONS = {}

for skill, aliases in TECH_SKILLS.items():

    for alias in [skill] + aliases:

        normalised = key(alias)

        if (
            normalised in ALIAS_TO_SKILL
            and ALIAS_TO_SKILL[normalised] != skill
        ):

            if normalised not in ALIAS_COLLISIONS:
                ALIAS_COLLISIONS[normalised] = set()

            ALIAS_COLLISIONS[normalised].add(
                ALIAS_TO_SKILL[normalised]
            )

            ALIAS_COLLISIONS[normalised].add(skill)

        else:
            ALIAS_TO_SKILL[normalised] = skill


# Canonical names take priority over aliases.

for skill in ALL_SKILLS:

    normalised = key(skill)

    ALIAS_TO_SKILL[normalised] = skill

    ALIAS_COLLISIONS.pop(normalised, None)


def recognise_skill(value, fuzzy=False):

    normalised = key(value)

    if not normalised:

        return {
            "input": str(value),
            "skill": None,
            "status": "empty",
            "suggestions": []
        }

    if normalised in ALIAS_COLLISIONS:

        return {
            "input": str(value),
            "skill": None,
            "status": "ambiguous",
            "suggestions": sorted(
                ALIAS_COLLISIONS[normalised]
            )
        }

    if normalised in ALIAS_TO_SKILL:

        return {
            "input": str(value),
            "skill": ALIAS_TO_SKILL[normalised],
            "status": "recognised",
            "suggestions": []
        }

    suggestions = []

    if fuzzy:

        matches = get_close_matches(
            normalised,
            list(ALIAS_TO_SKILL),
            n=3,
            cutoff=0.78
        )

        suggestions = list(
            dict.fromkeys(
                ALIAS_TO_SKILL[item]
                for item in matches
            )
        )

    return {
        "input": str(value),
        "skill": None,
        "status": "unrecognised",
        "suggestions": suggestions
    }


def parse_skills(value):

    if isinstance(value, (list, tuple, set)):
        parts = list(value)

    elif value is None:
        parts = []

    else:
        parts = re.split(
            r"[,;\n]+",
            str(value)
        )

    results = []

    for item in parts:

        if not str(item).strip():
            continue

        if key(item) in ["nan", "none"]:
            continue

        result = recognise_skill(item)

        if result["status"] == "recognised":
            skill = result["skill"]
        else:
            # Preserve unknown skills from the training data.
            skill = str(item).strip()

        if skill not in results:
            results.append(skill)

    return results


def encode_skills(value):

    return "|".join(
        parse_skills(value)
    )


def assess_input(value):

    if isinstance(value, (list, tuple, set)):
        parts = list(value)

    else:
        parts = re.split(
            r"[,;\n]+",
            str(value)
        )

    return [
        recognise_skill(item, fuzzy=True)
        for item in parts
        if str(item).strip()
    ]


def taxonomy_scores(skills):

    totals = {
        track: 0.0
        for track in TRACKS
    }

    for skill in set(skills):

        mappings = SKILL_SPECIALISATIONS.get(
            skill,
            {}
        )

        for track, weight in mappings.items():
            totals[track] += weight

    total = sum(totals.values())

    if total > 0:

        totals = {
            track: value / total
            for track, value in totals.items()
        }

    return totals


if __name__ == "__main__":

    print("Canonical skills:", len(ALL_SKILLS))

    print(
        "Alias collisions:",
        len(ALIAS_COLLISIONS)
    )

    examples = [
        "Packet Tracer",
        "data visualization",
        "machine learning",
        "powerbi",
        "Wireshark",
        "something unknown"
    ]

    for example in examples:

        print(
            example,
            "->",
            recognise_skill(example, fuzzy=True)
        )