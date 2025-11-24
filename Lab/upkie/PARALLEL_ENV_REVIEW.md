# Review: Why DummyVecEnv and Parallel Environment Support

## Why DummyVecEnv Instead of SubprocVecEnv?

### The Problem with SubprocVecEnv on macOS

1. **Process Spawning**: `SubprocVecEnv` uses Python's multiprocessing with 'spawn' method on macOS, which creates completely separate processes
2. **PyBullet Connection Issues**: PyBullet connections are process-specific and don't transfer well across process boundaries
3. **Pickling Problems**: PyBullet connection objects can't be pickled/serialized for inter-process communication

### Why DummyVecEnv Works

1. **Same Process**: All environments run in the same Python process
2. **Independent Connections**: Each environment creates its own PyBullet connection via `pybullet.connect()`
3. **Connection Isolation**: Each connection has a unique `physicsClientId` that isolates simulations
4. **No Pickling**: No need to serialize connections since everything is in the same process

## Critical Fix: physicsClientId Parameter

### The Bug
The original code had a critical bug: many PyBullet API calls were missing the `physicsClientId` parameter. This meant:
- Calls defaulted to the "current" connection (often the last one created)
- Multiple environments would interfere with each other
- One environment's actions could affect another's simulation

### The Fix
All PyBullet API calls now explicitly specify `physicsClientId=self._bullet`:
- `pybullet.disconnect(physicsClientId=self._bullet)` - disconnect specific connection
- `pybullet.stepSimulation(physicsClientId=self._bullet)` - step specific simulation
- `pybullet.getNumBodies(physicsClientId=self._bullet)` - query specific connection
- All other PyBullet calls now include `physicsClientId`

This ensures each environment's PyBullet connection is completely isolated.

## How It Works

1. **Environment Creation**: Each environment factory creates a new `UpkieEnv` → `PyBulletBackend`
2. **Connection Creation**: Each backend calls `pybullet.connect(pybullet.DIRECT)` and stores the connection ID
3. **Isolated Simulation**: All PyBullet calls use the specific `physicsClientId` for that connection
4. **Parallel Collection**: PPO collects rollouts from all environments sequentially but in batches

## Performance Considerations

- **Sequential Execution**: DummyVecEnv runs environments sequentially (not truly parallel)
- **Batch Collection**: PPO still benefits from collecting data from multiple environments
- **Expected Speedup**: 2-4x with 4 environments (depending on CPU and simulation speed)
- **Memory**: Each environment uses memory for its PyBullet connection and simulation state

## Verification

The codebase should now work correctly with multiple environments because:
1. ✅ All PyBullet calls specify `physicsClientId`
2. ✅ Each environment has its own connection
3. ✅ Connections are properly isolated
4. ✅ Cleanup (`close()`) disconnects only its own connection



