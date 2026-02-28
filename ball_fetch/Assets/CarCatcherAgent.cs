using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CarCatcherAgent : Agent
{
    [Header("References")]
    public Rigidbody rb;
    public Transform driveLeft;
    public Transform driveRight;
    public Transform basketCenter;
    public BallSpawner spawner;

    [Header("Drive Settings")]
    public float wheelForce = 25f;     // 左右轮推力
    public float maxSpeed = 8f;        // 限速，防止乱飞
    public float angularDamping = 0.2f;

    [Header("Episode Settings")]
    public float stepPenalty = -0.001f;
    public float distanceRewardScale = 0.002f; // 可选：越靠近篮子中心越好
    public float catchReward = 3.0f;
    public float missReward = -1.0f;
    public float arenaRadius = 6f;


    [Header("Stability")]
    public Vector3 centerOfMassLocal = new Vector3(0f, -0.25f, 0f);
    public float antiTipTorque = 0f; // 先不管，可选


    public bool debugNoEndEpisode = false;

    private Ball currentBall;

    public override void Initialize()
    {
        if (rb == null) rb = GetComponent<Rigidbody>();
        rb.maxAngularVelocity = 20f;
        rb.centerOfMass = centerOfMassLocal;
    }

    public override void OnEpisodeBegin()
    {
        // Reset car pose
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        Vector3 pos = new Vector3(
            Random.Range(-arenaRadius * 0.4f, arenaRadius * 0.4f),
            0.5f,
            Random.Range(-arenaRadius * 0.4f, arenaRadius * 0.4f)
        );
        transform.position = pos;
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

        // Spawn / reset ball
        currentBall = spawner.SpawnOrResetBall(this);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (currentBall == null)
        {
            // 占位，避免空引用
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(transform.forward);
            sensor.AddObservation(rb.linearVelocity);
            return;
        }

        Vector3 relPos = currentBall.transform.position - basketCenter.position;
        Vector3 relVel = currentBall.rb.linearVelocity - rb.linearVelocity;

        // 关键状态观测（建议先保留，训练快很多）
        sensor.AddObservation(relPos);                 // 3
        sensor.AddObservation(relVel);                 // 3
        sensor.AddObservation(transform.forward);      // 3
        sensor.AddObservation(rb.linearVelocity);            // 3
        // 总计 12 维（外加相机图像观测）
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Discrete: 0..4 => -2,-1,0,1,2
        int aL = actions.DiscreteActions[0];
        int aR = actions.DiscreteActions[1];
        float leftCmd  = (aL - 2) / 2f;   // -1..1
        float rightCmd = (aR - 2) / 2f;

        // 差速驱动：在左右驱动点施加前向力
        Vector3 fwd = transform.forward;
        rb.AddForceAtPosition(fwd * (leftCmd * wheelForce), driveLeft.position, ForceMode.Force);
        rb.AddForceAtPosition(fwd * (rightCmd * wheelForce), driveRight.position, ForceMode.Force);

        // 简单阻尼+限速，稳定训练
        rb.angularVelocity *= (1f - angularDamping * Time.fixedDeltaTime);

        Vector3 v = rb.linearVelocity;
        if (v.magnitude > maxSpeed)
            rb.linearVelocity = v.normalized * maxSpeed;

        // step penalty
        AddReward(stepPenalty);

        // 可选：密集奖励（球靠近篮子中心更好）
        if (currentBall != null)
        {
            float d = Vector3.Distance(currentBall.transform.position, basketCenter.position);
            AddReward(-d * distanceRewardScale);
        }
    }

    // 被 CatchZone 调用
    public void OnBallCaught()
    {
        AddReward(catchReward);
        if (!debugNoEndEpisode) EndEpisode();
    }

    // 被 Ball / Spawner 调用：球落地或超界
    public void OnBallMissed()
    {
        AddReward(missReward);
        if (!debugNoEndEpisode) EndEpisode();
    }

    // 方便你手动测试（WASD）
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var da = actionsOut.DiscreteActions;
        // 默认停
        da[0] = 2; // left wheel stop
        da[1] = 2; // right wheel stop

        // W/S 前后
        if (Input.GetKey(KeyCode.W)) { da[0] = 4; da[1] = 4; }
        if (Input.GetKey(KeyCode.S)) { da[0] = 0; da[1] = 0; }
        // A/D 原地转
        if (Input.GetKey(KeyCode.A)) { da[0] = 0; da[1] = 4; }
        if (Input.GetKey(KeyCode.D)) { da[0] = 4; da[1] = 0; }
    }
}
