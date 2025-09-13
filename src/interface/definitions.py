"""
Common technical term definitions for Theta AI.
This file contains accurate, concise definitions for technical terms to improve response quality.
"""

# Dictionary of technical term definitions
TECHNICAL_DEFINITIONS = {
    "api": "API stands for Application Programming Interface. It's a set of rules that allows different software applications to communicate with each other.",
    "rest api": "REST API (Representational State Transfer API) is an architectural style for designing networked applications. It uses HTTP requests to access and manipulate data, relying on standard methods like GET, POST, PUT, and DELETE.",
    "sdk": "SDK stands for Software Development Kit. It's a collection of software tools and libraries that developers use to create applications for specific platforms.",
    "ide": "IDE stands for Integrated Development Environment. It's a software application that provides comprehensive facilities for software development, typically including a code editor, debugger, and build automation tools.",
    "framework": "A framework is a pre-built structure or foundation of code that provides generic functionality which can be extended by developers to build applications.",
    "library": "A library is a collection of pre-written code that developers can use to perform common tasks without writing the code from scratch.",
    "algorithm": "An algorithm is a step-by-step procedure or formula for solving a problem, based on conducting a sequence of specified actions.",
    "database": "A database is an organized collection of structured information or data, typically stored electronically in a computer system.",
    "sql": "SQL (Structured Query Language) is a programming language used to manage and manipulate relational databases.",
    "nosql": "NoSQL (Not Only SQL) refers to non-relational databases designed for specific data models with flexible schemas for building modern applications.",
    "cloud computing": "Cloud computing is the delivery of computing services—including servers, storage, databases, networking, software, and analytics—over the internet.",
    "virtualization": "Virtualization is the creation of a virtual version of something, such as an operating system, a server, a storage device, or network resources.",
    "container": "A container is a lightweight, standalone executable package that includes everything needed to run a piece of software, including the code, runtime, system tools, libraries, and settings.",
    "docker": "Docker is a platform that uses containerization technology to make it easier to create, deploy, and run applications by using containers.",
    "kubernetes": "Kubernetes is an open-source container orchestration platform designed to automate deploying, scaling, and operating application containers.",
    "ci/cd": "CI/CD stands for Continuous Integration and Continuous Delivery/Deployment. It's a method to frequently deliver apps to customers by introducing automation into the stages of app development.",
    "git": "Git is a distributed version control system for tracking changes in source code during software development.",
    "firewall": "A firewall is a network security device or software that monitors and filters incoming and outgoing network traffic based on predetermined security rules.",
    "vpn": "VPN stands for Virtual Private Network. It creates a secure, encrypted connection over a less secure network, such as the internet.",
    "encryption": "Encryption is the process of converting information or data into a code to prevent unauthorized access.",
    "malware": "Malware is any software intentionally designed to cause damage to a computer, server, client, or computer network.",
    "phishing": "Phishing is a cybercrime in which targets are contacted by email, telephone, or text message by someone posing as a legitimate institution to lure them into providing sensitive data.",
    "ransomware": "Ransomware is a type of malicious software designed to block access to a computer system until a sum of money is paid.",
    "two-factor authentication": "Two-factor authentication (2FA) is a security process in which users provide two different authentication factors to verify their identity.",
    "ddos": "DDoS stands for Distributed Denial of Service. It's an attack where multiple compromised systems are used to target a single system causing a denial of service.",
    "blockchain": "Blockchain is a distributed database or ledger shared among computer network nodes that stores information electronically in digital format.",
    "cryptography": "Cryptography is the practice and study of techniques for secure communication in the presence of third parties.",
    "hash function": "A hash function is any function that can be used to map data of arbitrary size to fixed-size values, typically used to accelerate table lookup or data comparison.",
    "microservices": "Microservices is an architectural style that structures an application as a collection of small, loosely coupled services.",
    "devops": "DevOps is a set of practices that combines software development (Dev) and IT operations (Ops) to shorten the systems development life cycle.",
    "machine learning": "Machine learning is a branch of artificial intelligence focused on building applications that learn from data and improve their accuracy over time without being explicitly programmed.",
    "neural network": "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.",
    "agile": "Agile is an approach to software development that emphasizes iterative delivery, team collaboration, continual planning, and continual learning.",
    "scrum": "Scrum is an agile framework for developing, delivering, and sustaining complex products, with an emphasis on software development.",
    "waterfall": "Waterfall is a linear project management approach where stakeholder and customer requirements are gathered at the beginning of the project, and then a sequential project plan is created.",
    "defense in depth": "Defense in depth is a cybersecurity strategy that employs multiple layers of security controls throughout an IT system. If one security control fails, others still provide protection.",
    "penetration testing": "Penetration testing is the practice of testing a computer system, network, or web application to find security vulnerabilities that an attacker could exploit."
}

# Specific identity answers
IDENTITY_ANSWERS = {
    "who are you": "I'm Theta AI, an assistant developed by Frostline Solutions to help with cybersecurity, software development, and other technical topics. I was designed to provide accurate information and assistance to Frostline employees and clients.",
    "who created you": "I was created by Frostline Solutions, founded by Dakota Fryberger (CEO) and Devin Fox (Co-CEO). I'm designed to provide reliable information about cybersecurity, software development, and other technical topics.",
    "what are you": "I'm Theta AI, a specialized assistant focused on providing accurate information about cybersecurity, software development, and technical concepts. I combine retrieval-based and generative AI approaches to deliver helpful responses.",
    "tell me about yourself": "I'm Theta AI, a specialized assistant created by Frostline Solutions. I focus on cybersecurity, software development, and technical support topics. I'm trained on curated data to provide accurate and helpful information while avoiding hallucinations.",
    "how do you work": "I work through a hybrid approach that combines retrieval-based and generative AI techniques. When you ask a question, I first search through my knowledge base, then use a fine-tuned language model to generate a coherent, contextually appropriate response.",
    "what can you do": "I can answer questions about cybersecurity, software development, cloud computing, network security, hardware, and IT support. I can explain technical concepts, provide definitions, suggest best practices, and help troubleshoot common issues."
}

# Commonly hallucinated topics that should trigger safety responses
HALLUCINATION_PRONE_TOPICS = [
    "quantum computing",
    "blockchain consensus",
    "cryptographic protocols",
    "zero knowledge proofs", 
    "neural architecture",
    "quantum cryptography",
    "distributed ledger",
    "homomorphic encryption",
    "federated learning",
    "tensor calculus"
]

# Safety responses for potentially hallucination-prone topics
SAFETY_RESPONSES = {
    "limited_knowledge": "I have limited detailed information about this specific topic in my knowledge base. I can provide a general overview, but for in-depth or specialized information, please consult authoritative sources or documentation.",
    "technical_limitation": "This is a complex technical topic that requires precise information. While I can provide a basic explanation, please refer to official documentation or specialized resources for detailed implementation guidance.",
    "uncertainty": "I don't have enough specific information about this in my knowledge base to provide a complete answer. To avoid providing potentially inaccurate information, I recommend consulting official documentation or specialized resources."
}
