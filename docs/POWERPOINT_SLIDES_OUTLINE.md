# Monarch PowerPoint Presentation: Slide Content

## SECTION 1: HIGH-LEVEL PRESENTATION (15-20 slides)

---

### Slide 1: Title Slide
**Title:** Monarch: Distributed Programming Framework
**Subtitle:** Building Scalable Multi-Machine Training Systems
**Visual:** Monarch butterfly logo (if available), clean modern background
**Notes:** Opening slide - introduce speaker, set context

---

### Slide 2: What is Monarch?
**Main Points:**
- Distributed programming framework for PyTorch
- Based on scalable actor messaging
- Enables multi-machine training with clear semantics
- Built on actor model principles (similar to Erlang)

**Tagline:** "Simple, Scalable, Fault-Tolerant"

**Visual:** High-level system diagram showing multiple machines connected

---

### Slide 3: Four Major Features
**Feature 1: Scalable Messaging**
- Multidimensional meshes of actors
- Broadcast capabilities
- Organized communication patterns

**Feature 2: Fault Tolerance**
- Supervision trees
- Custom supervision handlers
- Automatic error propagation

**Feature 3: Point-to-Point RDMA**
- Low-level memory transfers
- Using libibverbs
- High-performance data transfer

**Feature 4: Distributed Tensors**
- Tensors sharded across processes
- Built-in support
- Transparent computation

**Visual:** 4-quadrant diagram with icons for each feature

---

### Slide 4: System Hierarchy Overview
**Three-Level Hierarchy:**
1. **HostMesh** (Blue) → Physical/Virtual Machines
   - Example: 32 hosts in a cluster

2. **ProcMesh** (Orange) → OS Processes
   - Example: 8 GPUs per host = 256 processes

3. **ActorMesh** (Green) → Actor Instances
   - Example: Multiple actors per process

**Visual:** Hierarchical diagram showing Host → Process → Actor layers
- Use color coding: Blue → Orange → Green

---

### Slide 5: Core Concepts - Part 1
**Actor**
- Isolated state machine
- Processes messages asynchronously
- Follows strict lifecycle semantics
- Example: counter increments value

**Mesh**
- Multidimensional container
- Organizes hosts, processes, or actors
- Named dimensions ("hosts", "gpus")
- Supports slicing operations

**Visual:** Simple actor box with private state, message queue, lifecycle arrows

---

### Slide 6: Core Concepts - Part 2
**Endpoint**
- Function decorated with @endpoint
- Defines public API of actor
- Remote callable methods
- Example: trainer.train(batch_size=32)

**Process & Host**
- Process: OS-level process hosting actors
- Host: Physical/virtual machine
- Multiple processes per host
- Multiple hosts per cluster

**Visual:** Actor with multiple @endpoint methods labeled

---

### Slide 7: Communication Patterns
**Messaging Adverbs (How to call actors):**

1. **call_one()** - Single actor, get response
   - Point-to-point communication

2. **call()** - All actors, collect responses
   - Broadcast and aggregate

3. **broadcast()** - Send to all, fire-and-forget
   - No response collection

4. **rref()** - Distributed tensor reference
   - For tensor operations

5. **stream()** - Stream responses as they arrive
   - Process incrementally

**Visual:** 5 boxes showing communication patterns with arrows

---

### Slide 8: Architecture Overview
**Complete System View:**
- HostMesh spawns ProcMesh
- ProcMesh spawns multiple ActorMeshes
- Each ActorMesh is independent
- Communication flows through defined patterns

**Key Relationships:**
- Parent-child relationships form supervision tree
- Fault propagation up the tree
- Custom handlers for error recovery

**Visual:** Mermaid-style hierarchy diagram with colored levels

---

### Slide 9: Supervision Tree
**Error Handling Model:**
- Similar to Erlang/OTP
- Each process/actor has parent supervisor
- Failures propagate up tree
- Custom __supervise__ method for recovery

**Benefits:**
- Automatic fault detection
- Graceful error handling
- System resilience
- Fine-grained control

**Visual:** Tree diagram showing supervisor nodes in red, child nodes in other colors, with error arrows

---

### Slide 10: Message Ordering Guarantees
**Strong Guarantees:**
1. **FIFO Ordering** - Messages from A→B in send order
2. **Sequential Processing** - Each actor processes one message at a time
3. **Concurrent Actors** - Different actors handle messages in parallel

**Benefits:**
- Predictable behavior
- No race conditions within actor
- Deterministic state management

**Visual:** Timeline showing messages M1, M2, M3 being processed sequentially

---

### Slide 11: RDMA Support
**What is RDMA?**
- Remote Direct Memory Access
- One-sided data transfers
- No remote CPU involvement needed
- High-performance data movement

**Use Cases:**
- Parameter server pulling weights
- Large tensor distribution
- Efficient gradient aggregation

**Architecture:**
- Separation of concerns: Control plane (messages) vs Data plane (RDMA)

**Visual:** Parameter server with RDMA arrows connecting to workers

---

### Slide 12: Distributed Tensors
**Key Features:**
- Automatic sharding across process meshes
- Transparent computation
- Integration with actor endpoints via rref()

**Benefits:**
- Seamless distributed computation
- No manual data placement
- Type-safe tensor operations

**Example Pattern:**
```
with procs.activate():
    t = torch.rand(3, 4)  # Automatically distributed
    result = t @ t.T      # Distributed computation
```

**Visual:** Tensor split across multiple processes with computation arrows

---

### Slide 13: Scenario 1 - Single Host, Multiple GPUs
**Setup:**
- Local development or single-machine training
- One host with 8 GPUs
- One process per GPU

**Code Flow:**
1. Get this_host()
2. spawn_procs(per_host={"gpus": 8})
3. spawn("trainers", Trainer)
4. train.call(batch=data)

**Hierarchy:**
```
this_host()
└── ProcMesh (gpus: 8)
    └── ActorMesh[Trainer] (8 instances)
```

**Visual:** Single machine with 8 GPU boxes stacked

---

### Slide 14: Scenario 2 - Multiple Hosts, Distributed
**Setup:**
- Large-scale distributed training
- 32 hosts × 8 GPUs per host = 256 processes
- Cluster-based deployment

**Code Flow:**
1. hosts_from_config("MONARCH_HOSTS")
2. spawn_procs(per_host={"gpus": 8})
3. spawn("trainers", Trainer)
4. train.call(step=0)  # All 256 trainers

**Scale:**
- All 256 trainers execute in parallel
- Automatic message routing across network
- Transparent failure handling

**Visual:** Multiple machines arranged in grid, each with multiple GPU boxes

---

### Slide 15: Scenario 3 - Actor-to-Actor Communication
**Pattern:**
- DataLoader provides data to Trainer
- Rank-based selection ensures correspondence
- Type-safe communication

**Benefits:**
- Modular design
- Each actor has clear responsibility
- Easy to reason about data flow

**Example Pattern:**
- 8 DataLoaders → 8 Trainers (1-to-1 mapping)
- Each trainer pulls from corresponding loader

**Visual:** DataLoader and Trainer pairs connected with bidirectional arrows

---

### Slide 16: Scenario 4 - RDMA Parameter Server
**Architecture:**
- 1 parameter server (holds model weights)
- 8 workers (pull parameters, compute gradients)
- RDMA for efficient weight transfer

**Flow:**
1. Worker pulls params via RDMA (fast)
2. Worker trains locally
3. Worker pushes gradients via message (slower OK)
4. Server updates weights

**Performance:**
- RDMA: Megabytes/Microsecond
- Messages: Used for small control data

**Visual:** Central server connected to 8 workers with two types of arrows (RDMA thick, Message thin)

---

### Slide 17: Mesh Slicing & Indexing
**What is Slicing?**
- Selecting subsets of mesh along dimensions
- Multidimensional indexing support
- Named dimensions for clarity

**Examples:**
```
first_gpu = actors.slice(gpus=0)
gpu_range = actors.slice(gpus=slice(0, 4))
subset = actors.slice(hosts=slice(0, 2), gpus=slice(0, 4))
```

**Use Cases:**
- Broadcast to subset of actors
- One-to-one communication
- Hierarchical patterns

**Visual:** Grid showing full mesh with colored highlight on sliced subset

---

### Slide 18: Benefits & Design Principles
**Why Use Monarch?**
✓ Clear Semantics - Explicit distribution patterns
✓ Type Safety - IDE autocomplete, runtime checks
✓ Fault Tolerant - Built-in supervision trees
✓ High Performance - RDMA, optimized messaging
✓ Location Transparent - Same code for local/remote

**Design Philosophy:**
- Inspired by Erlang's OTP actor model
- PyTorch-first focus
- Scalable from laptop to 1000+ machines

**Visual:** Checklist with checkmarks, or benefit icons

---

### Slide 19: When to Use Monarch
**Perfect For:**
- Distributed PyTorch training
- Multi-GPU/Multi-host systems
- Fault-tolerant applications
- Complex distributed patterns

**Not Ideal For:**
- Simple single-GPU training
- Synchronous batch processing
- Tightly-coupled computations

**Visual:** Venn diagram or comparison table

---

### Slide 20: Key Takeaways (High-Level)
1. **Actor Model** - Isolated, message-based computation
2. **Mesh Hierarchy** - Hosts → Processes → Actors
3. **Clear Communication** - 5 messaging patterns (call_one, call, broadcast, rref, stream)
4. **Fault Tolerance** - Supervision trees with custom handlers
5. **High Performance** - RDMA, distributed tensors, optimized messaging
6. **Developer Friendly** - Type-safe, location-transparent APIs

**Visual:** Icon or summary table for each point

---

---

## SECTION 2: LOW-LEVEL DEEP DIVE (20-25 slides)

---

### Slide 21: Actor Lifecycle - Detailed
**Five Stages:**

1. **Creation** - spawn() called
   - Runtime allocates resources
   - Actor registered
   - Mailbox created

2. **Construction** - __init__() method
   - Initialize state
   - Cannot send messages yet
   - No runtime services available

3. **Initialization** - init() hook (optional)
   - Runtime now available
   - Can spawn children
   - Can send/receive messages

4. **Running** - Process messages
   - Sequential processing
   - FIFO ordering
   - Handle requests

5. **Termination** - Cleanup
   - All children terminated
   - Resources released
   - Parent notified

**Visual:** State machine diagram with arrows between stages

---

### Slide 22: Actor Creation Phase
**Code Example:**
```python
from monarch.actor import Actor, endpoint, this_proc

class MyActor(Actor):
    def __init__(self, param1: int, param2: str):
        self.param1 = param1
        self.param2 = param2
        self.state = {}

# Spawn on current process
actor = this_proc().spawn("my_actor", MyActor, param1=42, param2="hello")
```

**What Happens:**
1. Runtime allocates resources
2. Actor is registered in system
3. Mailbox is created for message delivery
4. Unique Actor ID is assigned

**Important:** Actor ID is needed for routing messages across network

**Visual:** Code on left, process flow on right

---

### Slide 23: Message Processing Flow
**Step-by-Step:**
1. Message arrives in mailbox
2. Taken from queue (FIFO order)
3. Corresponding endpoint invoked
4. Handler method executes
5. Result computed or exception raised
6. Response sent back to sender

**Characteristics:**
- **Sequential** - One message at a time
- **Atomic** - No interleaving with other messages
- **Ordered** - FIFO from same sender

**Visual:** Mailbox queue with message flow diagram showing handler execution

---

### Slide 24: Messaging Adverbs - Detailed Comparison

**1. call_one() - Point-to-Point**
```python
result = actor.method.call_one(arg).get()
```
- Single actor only
- Wait for response
- Returns Future[Result]
- Use: One-to-one communication

**2. call() - Broadcast & Collect**
```python
results = actors.method.call(arg).get()
```
- All actors in mesh
- Wait for all responses
- Returns Future[List[Result]]
- Use: Parallel computation with aggregation

**3. broadcast() - Fire & Forget**
```python
actors.method.broadcast(arg)
```
- All actors in mesh
- No response collection
- Returns immediately
- Use: Side effects only, maximum throughput

**4. rref() - Distributed Tensor**
```python
with procs.activate():
    output = actor.forward.rref(input)
```
- Distributed tensor result
- Computation across actors
- Transparent sharding
- Use: Neural network layers

**5. stream() - Streaming**
```python
async for result in actors.method.stream(data):
    process(result)
```
- Process responses as they arrive
- Don't wait for all
- Incremental processing
- Use: Pipeline patterns

**Visual:** 5 comparison boxes showing flow diagrams for each

---

### Slide 25: call() Broadcast Flow
**Detailed Sequence:**

1. **Client initiates** - actors.method.call(args)
2. **Mesh routes** - Creates delivery tasks
3. **Parallel send** - All actors receive simultaneously
4. **Parallel execute** - Endpoints run concurrently on each actor
5. **Response collection** - Results stream back
6. **Aggregation** - Collected into list
7. **Return** - Future[List[Result]] to client

**Key Property:**
- All actors process in **true parallel**
- No blocking between actors
- Responses collected as they arrive

**Visual:** Timeline diagram showing parallel execution bars, then aggregation

---

### Slide 26: Actor Context API
**What is context()?**
- Runtime information about current execution
- Available inside endpoint methods
- Provides access to runtime services

**Three Main Components:**

1. **message_rank** - Position in mesh
```python
rank = context().message_rank
# {"hosts": 0, "gpus": 3}
```

2. **actor_instance** - Info about this actor
```python
inst = context().actor_instance
# actor_id, rank, proc_id
```

3. **proc** - Reference to hosting process
```python
proc = context().proc
# Can spawn siblings on same process
```

**Visual:** Context object with three branches showing available properties

---

### Slide 27: Using message_rank vs actor_rank
**Important Distinction:**

**actor_instance.rank** - Original position in full mesh
```python
# Actor's true position in ActorMesh
rank = context().actor_instance.rank
# {"hosts": 2, "gpus": 5}
```

**message_rank** - Position in sliced mesh (if called on slice)
```python
# If called on actors.slice(gpus=slice(0, 2))
msg_rank = context().message_rank
# {"hosts": 2, "gpus": 0 or 1}
```

**Use Case:**
- Use actor_rank for absolute positioning
- Use message_rank for relative positioning in slice

**Visual:** Two actors highlighted in different contexts - full mesh vs sliced mesh

---

### Slide 28: ActorMesh Structure
**What is ActorMesh?**
- Collection of actor instances
- Organized in multidimensional structure
- Inherits extent from ProcMesh
- Supports all messaging operations

**Properties:**
```python
actors = procs.spawn("trainers", Trainer)
print(type(actors))      # ActorMesh
print(actors.extent)     # {"hosts": 32, "gpus": 8}
print(actors.name)       # "trainers"
```

**Multidimensional Organization:**
- Each actor has unique coordinate
- Example: (host=2, gpu=5) → Actor instance
- Slicing works on these coordinates

**Visual:** 2D grid showing coordinate system, each cell is an actor

---

### Slide 29: Slicing Operations Deep Dive

**Single Index Slicing:**
```python
first_gpu = actors.slice(gpus=0)
# All hosts, only GPU 0
# extent: {"hosts": 32, "gpus": 1}
```

**Range Slicing:**
```python
first_four = actors.slice(gpus=slice(0, 4))
# All hosts, GPUs 0-3
# extent: {"hosts": 32, "gpus": 4}
```

**Multi-dimensional:**
```python
subset = actors.slice(hosts=slice(0, 2), gpus=slice(0, 4))
# extent: {"hosts": 2, "gpus": 4}
```

**Result:**
- Returns new ActorMesh with subset
- Can call endpoints on subset
- Type-safe indices

**Visual:** 3D grid showing various slicing operations with highlighted regions

---

### Slide 30: Three Types of Meshes - Deep Dive

**HostMesh (Level 1)**
- Represents physical/virtual machines
- Top of hierarchy
- Spawns ProcMesh
- Example extent: {"hosts": 32}

**ProcMesh (Level 2)**
- OS-level processes
- One per GPU typically
- Spawns ActorMesh
- Example extent: {"hosts": 32, "gpus": 8}
- Can activate for distributed tensors

**ActorMesh (Level 3)**
- Actor instances
- Multiple types can coexist
- Inherits parent dimensions
- Example: trainers, dataloaders, evaluators

**Hierarchy:**
HostMesh → spawn_procs() → ProcMesh → spawn() → ActorMesh

**Visual:** Three-level pyramid with extent annotations

---

### Slide 31: Dimension Inheritance
**How Dimensions Flow:**

1. **HostMesh** defines base dimensions
```python
hosts = hosts_from_config("MONARCH_HOSTS")
# extent: {"hosts": 32}
```

2. **ProcMesh** adds new dimensions
```python
procs = hosts.spawn_procs(per_host={"gpus": 8})
# extent: {"hosts": 32, "gpus": 8}
```

3. **ActorMesh** inherits exactly
```python
trainers = procs.spawn("trainers", Trainer)
dataloaders = procs.spawn("dataloaders", DataLoader)
# Both: {"hosts": 32, "gpus": 8}
```

**Key Principle:**
- Child inherits parent dimensions
- New dimensions added per level
- Consistent throughout hierarchy

**Visual:** Flow diagram with dimensions accumulating at each level

---

### Slide 32: Distributed Pattern 1 - Data Parallel
**Concept:**
- Each actor processes different data independently
- Results aggregated
- Simple and scalable

**Implementation:**
```python
class Trainer(Actor):
    @endpoint
    def train_step(self, step: int):
        batch = self.get_local_batch()
        loss = self.model(batch)
        return loss

trainers = procs.spawn("trainers", Trainer)
losses = trainers.train_step.call(step=0).get()
avg_loss = sum(losses) / len(losses)
```

**Characteristics:**
- All actors work simultaneously
- Communication only for aggregation
- Linear scalability

**Visual:** Multiple trainers with data flowing in, losses flowing out, aggregation arrow

---

### Slide 33: Distributed Pattern 2 - Parameter Server
**Concept:**
- Central server holds model parameters
- Workers pull parameters, compute gradients
- Server updates based on gradients

**Implementation:**
```python
ps = ps_procs.spawn("ps", ParameterServer)
workers = worker_procs.spawn("workers", Worker, ps)

# Worker code
class Worker(Actor):
    def __init__(self, ps):
        self.ps = ps.slice(gpus=0)  # Point to server

    @endpoint
    def train_step(self):
        params = self.ps.get_params.call_one().get()  # Pull
        grads = self.compute_gradients(params)
        self.ps.update_params.call_one(grads)  # Push
```

**Trade-offs:**
- Centralized state (easier for some algorithms)
- Server can become bottleneck
- Good for parameter synchronization

**Visual:** Central PS box with 8 worker boxes radiating, bidirectional arrows

---

### Slide 34: Distributed Pattern 3 - Pipeline Parallel
**Concept:**
- Model split into stages
- Each stage processes sequentially
- Different data flowing through stages

**Implementation:**
```python
class Stage1(Actor):
    @endpoint
    def forward(self, input):
        return self.layer1(input)

class Stage2(Actor):
    def __init__(self, stage1):
        self.stage1 = stage1.slice(**context().actor_instance.rank)

    @endpoint
    def forward(self, input):
        x = self.stage1.forward.call_one(input).get()
        return self.layer2(x)

# Create 3-stage pipeline
stage1 = procs.spawn("stage1", Stage1)
stage2 = procs.spawn("stage2", Stage2, stage1)
stage3 = procs.spawn("stage3", Stage3, stage2)
```

**Benefits:**
- Splits large model across machines
- Each stage on different hardware
- Pipelining for throughput

**Visual:** Linear flow Stage1 → Stage2 → Stage3 with data batches flowing through

---

### Slide 35: Distributed Pattern 4 - Hierarchical Communication
**Concept:**
- Two-level aggregation
- Workers on GPUs → Leaders aggregate per host → Final result
- Reduces network congestion

**Implementation:**
```python
class GroupLeader(Actor):
    def __init__(self, workers):
        rank = context().actor_instance.rank
        self.workers = workers.slice(hosts=rank["hosts"])

    @endpoint
    def aggregate(self):
        results = self.workers.compute.call().get()
        return sum(results) / len(results)

# All GPU workers
workers = gpu_procs.spawn("workers", Worker)

# One leader per host
leaders = host_procs.spawn("leaders", GroupLeader, workers)

# Two-level aggregation
group_results = leaders.aggregate.call().get()
final = sum(group_results) / len(group_results)
```

**Benefits:**
- Reduces network load
- Scales to large clusters
- Local aggregation improves latency

**Visual:** Tree structure showing workers → leaders → root aggregation

---

### Slide 36: Advanced Pattern - Explicit Response Ports
**Concept:**
- Asynchronous response handling
- Out-of-order responses
- Background processing

**Use Case:**
- Heavy computation that doesn't block
- Multiple responses to same call
- Advanced control flow

**Implementation:**
```python
from monarch.actor import Port

class AsyncProcessor(Actor):
    @endpoint(explicit_response_port=True)
    def process(self, port: Port[str], data: str):
        # Queue work
        self.queue.put((port, data))

    def _process_loop(self):
        while True:
            port, data = self.queue.get()
            result = self._heavy_computation(data)
            port.send(result)  # Send when ready
```

**Visual:** Queue diagram showing work being processed asynchronously and responses sent back

---

### Slide 37: Advanced Pattern - Supervision
**Concept:**
- Custom error handling
- Fine-grained recovery
- Failure propagation control

**Implementation:**
```python
class SupervisorActor(Actor):
    def __supervise__(self, event):
        print(f"Supervision event: {event}")

        if event.is_recoverable():
            self.restart_child(event.actor_id)
            return True  # Handled
        else:
            return False  # Propagate to parent
```

**Return Value:**
- True: Error handled, continue
- False: Propagate to parent's supervisor

**Visual:** Error propagation tree showing which errors are caught at each level

---

### Slide 38: Advanced Pattern - Channels
**Concept:**
- Low-level messaging
- Bidirectional communication
- Direct channels between actors

**Implementation:**
```python
from monarch.actor import Channel, Port

class Producer(Actor):
    @endpoint
    def register_consumer(self, port: Port):
        self.consumers.append(port)

    @endpoint
    def produce(self, data):
        for port in self.consumers:
            port.send(data)

class Consumer(Actor):
    def __init__(self, producer):
        self.port, self.receiver = Channel.open()
        producer.register_consumer.call_one(self.port)

    @endpoint
    async def consume(self):
        data = await self.receiver.recv()
        return data
```

**Visual:** Channel connection between Producer and Consumer boxes

---

### Slide 39: Best Practice - Actor Design
**DO ✓**
- Keep actors focused (single responsibility)
- Use immutable messages
- Handle errors gracefully
- Document endpoints

**DON'T ✗**
- Share mutable state between actors
- Block in endpoints (use async)
- Ignore supervision events
- Create circular dependencies

**State Management:**
```python
class GoodActor(Actor):
    def __init__(self):
        self.counter = 0  # All state in __init__
        self.data = []

    @endpoint
    def update(self, value):
        self.counter += 1
        self.data.append(value)
        return self.counter
```

**Visual:** Two columns - DO's with checkmarks, DON'Ts with X marks

---

### Slide 40: Best Practice - Endpoint Design
**Principles:**
- Clear, typed signatures
- Well-documented parameters
- Explicit return types
- Error handling

**Example:**
```python
@endpoint
def process_batch(self, batch_id: int, data: list[float]) -> dict:
    """
    Process a batch of data.

    Args:
        batch_id: Unique batch identifier
        data: List of data points

    Returns:
        Dictionary with processing results
    """
    results = self._process(data)
    return {
        "batch_id": batch_id,
        "results": results,
        "processed_at": time.time()
    }
```

**Type Safety:**
- IDE autocomplete
- Runtime type checking
- Documentation generation

**Visual:** Code example with annotations

---

### Slide 41: Best Practice - Error Handling
**Strategy:**
- Catch known errors locally
- Log and return error status
- Let unexpected errors propagate

**Implementation:**
```python
@endpoint
def risky_operation(self, data):
    try:
        result = self._process(data)
        return {"success": True, "result": result}
    except ValueError as e:
        logger.error(f"Invalid data: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise  # Let supervision handle it
```

**Visual:** Error flow diagram showing caught errors vs propagated errors

---

### Slide 42: Testing Actors
**Unit Testing Pattern:**
```python
import pytest
from monarch.actor import this_proc

@pytest.mark.asyncio
async def test_calculator():
    # Spawn actor
    calc = this_proc().spawn("test_calc", Calculator)

    # Test endpoint
    result = await calc.add.call_one(5, 3)
    assert result == 8

    # Test state
    history = await calc.get_history.call_one()
    assert len(history) == 1
    assert history[0] == ("add", 5, 3, 8)
```

**Testing Mesh:**
```python
@pytest.fixture
def test_mesh():
    procs = this_host().spawn_procs(per_host={"gpus": 2})
    yield procs
    procs.terminate()

def test_broadcast(test_mesh):
    actors = test_mesh.spawn("test", TestActor)
    results = actors.test_method.call(42).get()
    assert len(results) == 2
```

**Visual:** Test code with assertions highlighted

---

### Slide 43: Performance Optimization Tips

**Tip 1: Pre-allocate Resources**
```python
def __init__(self):
    self.buffer = torch.zeros(1000, 1000)
    self.cache = {}
```

**Tip 2: Reuse Buffers**
```python
@endpoint
def compute(self, input_data):
    self.buffer.copy_(input_data)
    result = self._compute(self.buffer)
    return result
```

**Tip 3: Cache Expensive Computations**
```python
if data_id in self.cache:
    return self.cache[data_id]
result = expensive_compute(data)
self.cache[data_id] = result
return result
```

**Tip 4: Use RDMA for Large Data**
- RDMA for >1MB transfers
- Messages for control data
- Separate control and data planes

**Visual:** Performance optimization checklist with code snippets

---

### Slide 44: Summary - Low Level Concepts
**Key Technical Points:**

1. **Lifecycle** - 5 stages: Creation, Construction, Init, Running, Termination
2. **Message Processing** - Sequential, FIFO, atomic
3. **Messaging Adverbs** - 5 patterns for different communication styles
4. **Context** - Runtime info via message_rank, actor_instance, proc
5. **ActorMesh** - Multidimensional, sliceable, broadcasting
6. **Three Meshes** - HostMesh, ProcMesh, ActorMesh with dimension inheritance
7. **Patterns** - Data parallel, Parameter server, Pipeline, Hierarchical
8. **Advanced** - Response ports, Supervision, Channels
9. **Best Practices** - Design, Testing, Error handling, Performance

**Visual:** Grid or table of concepts with brief descriptions

---

### Slide 45: Monarch Ecosystem
**Integrated Technologies:**
- PyTorch for neural networks
- RDMA for high-speed networking
- Supervision trees from Erlang/OTP
- Multidimensional mesh organization
- Distributed tensor computation

**Build On Monarch For:**
- Large-scale training systems
- Multi-GPU orchestration
- Fault-tolerant applications
- High-performance distributed algorithms

**Visual:** Monarch at center with connected technology boxes

---

### Slide 46: Next Steps & Resources
**To Get Started:**
1. Read MONARCH_OVERVIEW.md
2. Try examples from docs/examples/
3. Run on single host first
4. Scale to multiple hosts
5. Integrate with your training code

**Documentation:**
- docs/MONARCH_OVERVIEW.md - High-level intro
- docs/ACTORS.md - Deep dive into actors
- docs/MESHES.md - Deep dive into meshes
- docs/README_CONCEPTS.md - Navigation guide

**Visual:** Roadmap or learning path diagram

---

### Slide 47: Questions & Discussion
**Common Questions:**
- "How does Monarch compare to [other framework]?"
- "What's the learning curve?"
- "How do I migrate existing code?"
- "What are the performance characteristics?"

**Key Takeaways:**
- Actor model is proven, reliable
- Monarch brings OTP-style robustness to Python/PyTorch
- Scales from laptop to thousands of machines
- Type-safe, location-transparent API

**Visual:** Blank with "Questions?" or discussion format

---
