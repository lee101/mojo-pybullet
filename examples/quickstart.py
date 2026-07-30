import mojo_pybullet as p

client = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.8, physicsClientId=client)

floor_shape = p.createCollisionShape(
    p.GEOM_PLANE, planeNormal=(0, 0, 1), physicsClientId=client
)
ball_shape = p.createCollisionShape(
    p.GEOM_SPHERE, radius=0.5, physicsClientId=client
)
p.createMultiBody(0, floor_shape, physicsClientId=client)
ball = p.createMultiBody(
    1, ball_shape, basePosition=(0, 0, 2), physicsClientId=client
)

for _ in range(240):
    p.stepSimulation(physicsClientId=client)

position, _ = p.getBasePositionAndOrientation(ball, physicsClientId=client)
print(position)
print(p.rayTest((0, 0, 3), (0, 0, -1), physicsClientId=client))
p.disconnect(client)
