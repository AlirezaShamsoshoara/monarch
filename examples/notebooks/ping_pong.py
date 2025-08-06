import asyncio

from monarch.actor import Actor, current_rank, endpoint, proc_mesh, ProcMesh

NUM_ACTORS = 4


class ToyActor(Actor):
    def __init__(self):
        self.rank = current_rank().rank

    @endpoint
    async def hello_world(self, msg):
        print(f"Identity: {self.rank}, {msg=}")


# # Note: Meshes can be also be created on different nodes, but we're ignoring that in this example
# local_proc_mesh = await proc_mesh(gpus=NUM_ACTORS)
# # This spawns 4 instances of 'ToyActor'
# toy_actor = await local_proc_mesh.spawn("toy_actor", ToyActor)


# # Once actors are spawned, we can call all of them simultaneously with `Actor.endpoint.call` as below
# await toy_actor.hello_world.call("hey there, from jupyter!!")

# # We can also specify a single actor using the 'slice' API
# futures = []
# for idx in range(NUM_ACTORS):
#     actor_instance = toy_actor.slice(gpus=idx)
#     futures.append(
#         actor_instance.hello_world.call_one(f"Here's an arbitrary unique value: {idx}")
#     )

# # conveniently, we can still schedule & gather them in parallel using asyncio
# await asyncio.gather(*futures)


async def main():
    # Create a processing mesh
    local_proc_mesh = await proc_mesh(gpus=NUM_ACTORS)
    # Spawn 4 instances of 'ToyActor'
    toy_actor = await local_proc_mesh.spawn("toy_actor", ToyActor)

    # Call all actors simultaneously
    await toy_actor.hello_world.call("hey there, from jupyter!!")

    # Use the 'slice' API to specify a single actor
    futures = []
    for idx in range(NUM_ACTORS):
        actor_instance = toy_actor.slice(gpus=idx)
        futures.append(
            actor_instance.hello_world.call_one(
                f"Here's an arbitrary unique value: {idx}"
            )
        )

    # Schedule & gather them in parallel
    await asyncio.gather(*futures)


# Run the main function
asyncio.run(main())


# import asyncio

# from monarch.actor import Actor, current_rank, endpoint, proc_mesh, ProcMesh


# class ExampleActor(Actor):
#     def __init__(self, actor_name):
#         self.actor_name = actor_name

#     @endpoint
#     async def init(self, other_actor):
#         self.other_actor = other_actor
#         self.other_actor_pair = other_actor.slice(**current_rank())
#         self.identity = current_rank().rank

#     @endpoint
#     async def send(self, msg):
#         await self.other_actor_pair.recv.call(
#             f"Sender ({self.actor_name}:{self.identity}) {msg=}"
#         )

#     @endpoint
#     async def recv(self, msg):
#         print(f"Pong!, Receiver ({self.actor_name}:{self.identity}) received msg {msg}")


# # Spawn two different Actors in different meshes, with two instances each
# local_mesh_0 = await proc_mesh(gpus=2)
# actor_0 = await local_mesh_0.spawn(
#     "actor_0", ExampleActor, "actor_0"  # this arg is passed to ExampleActor.__init__
# )

# local_mesh_1 = await proc_mesh(gpus=2)
# actor_1 = await local_mesh_1.spawn(
#     "actor_1", ExampleActor, "actor_1"  # this arg is passed to ExampleActor.__init__
# )

# await asyncio.gather(
#     actor_0.init.call(actor_1),
#     actor_1.init.call(actor_0),
# )


# await actor_0.send.call("Ping")
# await actor_1.send.call("Ping")
