# Monarch Concepts Documentation

Welcome to the Monarch concepts documentation! This collection of documents provides a comprehensive guide to understanding and using the Monarch distributed programming framework.

## 📚 Documentation Structure

This documentation is organized into three main files, each covering different aspects of the Monarch framework:

### 1. [Overview](./MONARCH_OVERVIEW.md) - Start Here! 🚀

**Purpose:** High-level introduction to Monarch and its core concepts.

**Contents:**
- Introduction to Monarch
- Four major features (Messaging, Fault Tolerance, RDMA, Distributed Tensors)
- Core concepts (Actor, Mesh, Endpoint, Process, Host)
- System architecture and hierarchy
- Common scenarios and patterns
- Quick reference guide

**Best for:** New users, understanding the big picture, getting started

---

### 2. [Actors](./ACTORS.md) - Deep Dive into Actors 🎭

**Purpose:** Comprehensive guide to actors - the fundamental computation units.

**Contents:**
- What is an Actor?
- Complete actor lifecycle
- Endpoints and messaging adverbs
- Actor context and runtime information
- ActorMesh organization
- Advanced patterns (ports, channels, supervision)
- Best practices and testing

**Best for:** Understanding actor behavior, implementing actors, advanced patterns

---

### 3. [Meshes](./MESHES.md) - Deep Dive into Meshes 🌐

**Purpose:** Complete guide to all mesh types and their usage.

**Contents:**
- What is a Mesh?
- Three mesh types (HostMesh, ProcMesh, ActorMesh)
- Mesh hierarchy and relationships
- Operations (slicing, broadcasting, selection)
- Distributed patterns (data parallel, parameter server, pipeline)
- Best practices and testing

**Best for:** Understanding mesh organization, distributed patterns, resource management

---

## 🎯 Quick Navigation

### By Learning Goal

| Goal | Start Here |
|------|-----------|
| **I'm new to Monarch** | [Overview](./MONARCH_OVERVIEW.md) |
| **I want to build an actor** | [Actors](./ACTORS.md) → What is an Actor? |
| **I need to understand messaging** | [Actors](./ACTORS.md) → Endpoints and Messaging |
| **I'm setting up distributed training** | [Meshes](./MESHES.md) → Distributed Patterns |
| **I need to organize resources** | [Meshes](./MESHES.md) → Types of Meshes |
| **I want to see examples** | [Overview](./MONARCH_OVERVIEW.md) → Common Scenarios |
| **I need API reference** | [Overview](./MONARCH_OVERVIEW.md) → Quick Reference |

### By Concept

| Concept | Primary Location | Supporting Info |
|---------|-----------------|-----------------|
| **Actor** | [Actors](./ACTORS.md) | [Overview - Core Concepts](./MONARCH_OVERVIEW.md#core-concepts) |
| **Endpoint** | [Actors](./ACTORS.md#endpoints-and-messaging) | [Overview - Core Concepts](./MONARCH_OVERVIEW.md#3-endpoint) |
| **Mesh** | [Meshes](./MESHES.md) | [Overview - Core Concepts](./MONARCH_OVERVIEW.md#2-mesh) |
| **HostMesh** | [Meshes](./MESHES.md#1-hostmesh) | [Overview - Key Components](./MONARCH_OVERVIEW.md#1-hostmesh) |
| **ProcMesh** | [Meshes](./MESHES.md#2-procmesh) | [Overview - Key Components](./MONARCH_OVERVIEW.md#2-procmesh) |
| **ActorMesh** | [Actors](./ACTORS.md#actormesh) | [Meshes - ActorMesh](./MESHES.md#3-actormesh) |
| **Context** | [Actors](./ACTORS.md#actor-context) | [Overview - Key Components](./MONARCH_OVERVIEW.md#6-context) |
| **Future** | [Actors](./ACTORS.md#endpoints-and-messaging) | [Overview - Key Components](./MONARCH_OVERVIEW.md#5-future) |
| **Supervision** | [Actors](./ACTORS.md#advanced-patterns) | [Overview - Supervision Tree](./MONARCH_OVERVIEW.md#supervision-tree) |
| **RDMA** | [Overview](./MONARCH_OVERVIEW.md#rdma-support) | [Overview - Scenario 4](./MONARCH_OVERVIEW.md#scenario-4-rdma-based-parameter-server) |
| **Distributed Tensors** | [Overview](./MONARCH_OVERVIEW.md#distributed-tensors) | [Overview - Four Major Features](./MONARCH_OVERVIEW.md#four-major-features) |

---

## 📖 Reading Paths

### Path 1: Complete Beginner

1. Read [Overview - Introduction](./MONARCH_OVERVIEW.md#introduction)
2. Understand [Overview - Core Concepts](./MONARCH_OVERVIEW.md#core-concepts)
3. Review [Overview - Architecture](./MONARCH_OVERVIEW.md#architecture-overview)
4. Study [Actors - What is an Actor?](./ACTORS.md#what-is-an-actor)
5. Learn [Actors - Lifecycle](./ACTORS.md#actor-lifecycle)
6. Explore [Overview - Common Scenarios](./MONARCH_OVERVIEW.md#common-scenarios)
7. Try code examples from [Overview - Scenarios](./MONARCH_OVERVIEW.md#common-scenarios)

### Path 2: Building Distributed Training

1. Read [Meshes - What is a Mesh?](./MESHES.md#what-is-a-mesh)
2. Understand [Meshes - Types of Meshes](./MESHES.md#types-of-meshes)
3. Study [Meshes - Hierarchy](./MESHES.md#mesh-hierarchy)
4. Learn [Meshes - Operations](./MESHES.md#mesh-operations)
5. Review [Meshes - Distributed Patterns](./MESHES.md#distributed-patterns)
6. Implement [Meshes - Pattern 1: Data Parallel](./MESHES.md#pattern-1-simple-data-parallel)
7. Scale up with [Overview - Scenario 2](./MONARCH_OVERVIEW.md#scenario-2-multiple-hosts-multiple-processes)

### Path 3: Advanced Actor Patterns

1. Review [Actors - Actor Lifecycle](./ACTORS.md#actor-lifecycle)
2. Master [Actors - Endpoints and Messaging](./ACTORS.md#endpoints-and-messaging)
3. Learn [Actors - Actor Context](./ACTORS.md#actor-context)
4. Study [Actors - ActorMesh](./ACTORS.md#actormesh)
5. Implement [Actors - Advanced Patterns](./ACTORS.md#advanced-patterns)
6. Apply [Actors - Best Practices](./ACTORS.md#best-practices)
7. Test with [Actors - Testing](./ACTORS.md#best-practices)

### Path 4: Understanding the System

1. Study [Overview - System Hierarchy](./MONARCH_OVERVIEW.md#system-hierarchy)
2. Understand [Meshes - Mesh Hierarchy](./MESHES.md#mesh-hierarchy)
3. Learn [Actors - Actor Lifecycle](./ACTORS.md#actor-lifecycle)
4. Review [Overview - Supervision Tree](./MONARCH_OVERVIEW.md#supervision-tree)
5. Explore [Overview - Message Ordering](./MONARCH_OVERVIEW.md#message-ordering-guarantees)
6. Master [Actors - Messaging Adverbs](./ACTORS.md#messaging-adverbs)

---

## 🔍 Finding What You Need

### Common Questions

**"How do I create actors?"**
→ [Actors - Actor Lifecycle - Creation Phase](./ACTORS.md#1-creation-phase)

**"What's the difference between call() and broadcast()?"**
→ [Actors - Messaging Adverbs](./ACTORS.md#messaging-adverbs)

**"How do I organize my distributed system?"**
→ [Meshes - Types of Meshes](./MESHES.md#types-of-meshes)

**"How do I select a subset of actors?"**
→ [Meshes - Slicing](./MESHES.md#1-slicing)

**"What is context() and when do I use it?"**
→ [Actors - Actor Context](./ACTORS.md#actor-context)

**"How do I implement data parallel training?"**
→ [Meshes - Pattern 1: Data Parallel](./MESHES.md#pattern-1-simple-data-parallel)

**"How do I pass actors as arguments?"**
→ [Actors - ActorMesh - Passing References](./ACTORS.md#passing-actor-references)

**"What are the different mesh types?"**
→ [Meshes - Types of Meshes](./MESHES.md#types-of-meshes)

**"How does error handling work?"**
→ [Overview - Supervision Tree](./MONARCH_OVERVIEW.md#supervision-tree)

**"How do I use RDMA?"**
→ [Overview - RDMA Support](./MONARCH_OVERVIEW.md#rdma-support)

---

## 📊 Visual Guides

All three documents include comprehensive Mermaid diagrams:

### Architecture Diagrams
- System hierarchy
- Communication flows
- Mesh structures
- Component relationships

### Lifecycle Diagrams
- Actor lifecycle states
- Message processing flows
- Spawning sequences

### Pattern Diagrams
- Data parallel
- Parameter server
- Pipeline parallel
- Hierarchical communication

---

## 🎨 Color Coding in Diagrams

Throughout the documentation, diagrams use consistent color coding:

- 🔵 **Blue (#e1f5ff)**: HostMesh / Host-level
- 🟠 **Orange (#fff4e1)**: ProcMesh / Process-level
- 🟢 **Green (#e8f5e9)**: ActorMesh / Actor-level
- 🔴 **Red (#ff6b6b)**: Supervisors / Leaders
- 🟡 **Yellow (#ffe66d)**: Workers / Data

---

## 🛠️ Code Examples

Each document includes runnable code examples:

### Overview
- Basic actor creation
- Mesh spawning
- Common scenarios (1-4)
- Quick reference patterns

### Actors
- Complete actor implementations
- All messaging adverbs
- Context usage
- Advanced patterns
- Testing examples

### Meshes
- Creating all mesh types
- Slicing operations
- Distributed patterns
- Best practices
- Testing meshes

---

## 📝 Document Features

### Comprehensive Coverage
- **Overview**: 500+ lines covering all major concepts
- **Actors**: 800+ lines with deep dive into actor system
- **Meshes**: 700+ lines covering all mesh types

### Rich Diagrams
- 30+ Mermaid diagrams across all documents
- Architecture, sequence, and state diagrams
- Pattern visualizations

### Code Examples
- 50+ code examples
- Full implementations
- Best practices
- Anti-patterns (what NOT to do)

### Quick References
- Summary tables
- Comparison charts
- Command cheat sheets
- Common patterns

---

## 🚀 Getting Started

### For Absolute Beginners
1. Start with [Overview](./MONARCH_OVERVIEW.md)
2. Read sections in order
3. Run examples from [Common Scenarios](./MONARCH_OVERVIEW.md#common-scenarios)
4. Deep dive into [Actors](./ACTORS.md) or [Meshes](./MESHES.md) as needed

### For Experienced Developers
1. Skim [Overview](./MONARCH_OVERVIEW.md) for architecture
2. Jump to [Actors - Advanced Patterns](./ACTORS.md#advanced-patterns)
3. Review [Meshes - Distributed Patterns](./MESHES.md#distributed-patterns)
4. Implement your use case

### For System Architects
1. Study [Overview - System Hierarchy](./MONARCH_OVERVIEW.md#system-hierarchy)
2. Understand [Meshes - Mesh Hierarchy](./MESHES.md#mesh-hierarchy)
3. Review [Overview - Supervision Tree](./MONARCH_OVERVIEW.md#supervision-tree)
4. Plan your distributed system

---

## 📚 Additional Resources

### Official Documentation
- [Getting Started Tutorial](./source/examples/getting_started.py)
- [API Reference](./api/)
- [Examples Directory](../examples/)
- [Main README](../README.md)

### Related Concepts
- Hyperactor runtime (see [hyperactor-book](./source/books/hyperactor-book/))
- Distributed tensors
- RDMA operations

---

## 🤝 Contributing

If you find errors or have suggestions for improving this documentation:
1. See [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Open an issue describing the improvement
3. Submit a pull request with changes

---

## 📄 Document Status

- **Created**: 2024
- **Status**: Complete
- **Version**: 1.0
- **Coverage**: Comprehensive guide to all Monarch concepts

---

## 🎯 Key Takeaways

After reading these documents, you should understand:

✅ What actors are and how they work
✅ The three types of meshes and their relationships
✅ How to organize distributed systems with meshes
✅ Message passing and communication patterns
✅ Supervision trees and fault tolerance
✅ RDMA and distributed tensor capabilities
✅ Best practices for building robust systems

---

## 💡 Tips for Learning

1. **Start Simple**: Begin with single-host examples
2. **Run Examples**: Type out and run code samples
3. **Draw Diagrams**: Sketch your system architecture
4. **Test Often**: Write tests as you learn
5. **Ask Questions**: Refer back to documentation frequently
6. **Build Projects**: Apply concepts to real problems

---

Happy coding with Monarch! 🦋
