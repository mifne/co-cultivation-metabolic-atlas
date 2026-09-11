# Figure 1: AI-Driven Microbial Consortium for Rubber Degradation and PHA Synthesis

This conceptual diagram illustrates the integration of Reinforcement Learning (RL) with a genetically engineered three-species microbial consortium (Actinoplanes sp. OR16, Rhizobacter gummiphilus NS21, and Lactobacillus plantarum). 

```mermaid
graph TD
    %% Define Styles
    classDef ai fill:#f9f,stroke:#333,stroke-width:2px,color:#000,font-weight:bold
    classDef environment fill:#eef,stroke:#333,stroke-width:2px,color:#000
    classDef microbe fill:#d5f5e3,stroke:#2ecc71,stroke-width:2px,color:#000
    classDef metabolite fill:#fef9e7,stroke:#f1c40f,stroke-width:1px,color:#000
    classDef product fill:#fdedec,stroke:#e74c3c,stroke-width:2px,color:#000,font-weight:bold

    %% RL Agent Component
    subgraph RL["AI Control System (PPO Agent)"]
        Policy["Policy Network"]:::ai
        Reward["Reward Function<br/>(Maximize PHA, Minimize Cost, Survive)"]:::ai
    end

    %% Bioreactor Environment
    subgraph Bioreactor["Continuous Bioreactor (dFBA Environment)"]
        
        %% Actuators (Feeds)
        Feed_OR16["Feed: OR16 (Glucose/AA)"]
        Feed_NS21["Feed: NS21 (Glucose/AA)"]
        Feed_LP["Feed: LP (Glucose/AA)"]
        Feed_Common["Common Carbon Source"]
        Aeration["Aeration Control (kLa)"]

        %% Microbes
        OR16["Actinoplanes sp. OR16<br/>(Lcp: Endocleavage)"]:::microbe
        NS21["Rhizobacter gummiphilus NS21<br/>(Rox: Exocleavage + PHA Storage)"]:::microbe
        LP["Lactobacillus plantarum<br/>(pH Buffer & Stabilizer)"]:::microbe

        %% Metabolites / Physics
        Rubber["Polyisoprene (Rubber)"]:::metabolite
        C30["C30 Oligomers"]:::metabolite
        ODTD["ODTD (Oligo-tetradecadiene)"]:::metabolite
        PHA["Polyhydroxyalkanoates (PHA)"]:::product
        LacticAcid["Lactic Acid (pH Reducer)"]:::metabolite
        pH["Environmental pH"]:::metabolite

        %% Metabolic Handoff Flow
        Rubber -- "Lcp Enzyme" --> C30
        OR16 -. "Secretes" .-> C30
        C30 -- "Rox Enzyme" --> ODTD
        ODTD -- "Beta-Oxidation" --> PHA
        NS21 -. "Synthesizes" .-> PHA
        
        %% pH stabilization flow
        LP -. "Produces" .-> LacticAcid
        LacticAcid --> pH
        pH -. "Feedback loop" .-> LP
        
        %% Feed flows
        Feed_OR16 --> OR16
        Feed_NS21 --> NS21
        Feed_LP --> LP
    end

    %% Closed-Loop Interaction
    State["State Observation<br/>(Biomass, pH, O2, Metabolites)"]:::environment
    Action["Action<br/>(Feed Rates, Aeration)"]:::environment

    Bioreactor --> State
    State --> Policy
    Policy --> Action
    Action --> Feed_OR16
    Action --> Feed_NS21
    Action --> Feed_LP
    Action --> Feed_Common
    Action --> Aeration

```

**Caption:** 
**(A)** The Bioreactor environment simulates the dynamics of three interdependent bacterial species using dynamic Flux Balance Analysis (dFBA). *Actinoplanes sp. OR16* initiates rubber degradation by secreting Lcp, producing C30 oligomers. *Rhizobacter gummiphilus NS21* further breaks down these oligomers into ODTD and utilizes the carbon flux to accumulate Polyhydroxyalkanoates (PHA). *Lactobacillus plantarum* acts as a metabolic buffer, secreting lactic acid to counter the pH increases caused by rubber degradation. 
**(B)** The Proximal Policy Optimization (PPO) agent observes the real-time internal state of the bioreactor (biomass concentrations, pH, and metabolite levels) and continuously outputs actions (specific feed rates and aeration) to maximize PHA production while maintaining consortium survival and minimizing resource costs over a 28-day cycle.
