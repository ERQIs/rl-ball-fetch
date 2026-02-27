using UnityEngine;

public class BallSpawner : MonoBehaviour
{
    public Ball ballPrefab;

    [Header("Spawn")]
    public float spawnHeight = 3f;
    public float minForwardDist = 6f;
    public float maxForwardDist = 10f;
    public float sideJitter = 2f;

    [Header("Flight Time Range")]
    public float Tmin = 1.0f;
    public float Tmax = 1.8f;

    [Header("Landing Noise")]
    public float landingNoise = 0.4f;

    private Ball ballInstance;

    public Ball SpawnOrResetBall(CarCatcherAgent agent)
    {
        if (ballInstance == null)
            ballInstance = Instantiate(ballPrefab);

        ballInstance.agent = agent;

        // ===== 起点 =====
        Vector3 carPos = agent.transform.position;
        Vector3 forward = agent.transform.forward;

        float fwdDist = Random.Range(minForwardDist, maxForwardDist);
        Vector3 spawnPos = carPos + forward * fwdDist;
        spawnPos += agent.transform.right * Random.Range(-sideJitter, sideJitter);
        spawnPos.y = spawnHeight;

        ballInstance.transform.position = spawnPos;
        ballInstance.rb.linearVelocity = Vector3.zero;
        ballInstance.rb.angularVelocity = Vector3.zero;

        // ===== 落点（篮子中心 + 小随机）=====
        Vector3 target = agent.basketCenter.position;
        target += agent.transform.right * Random.Range(-landingNoise, landingNoise);
        target += agent.transform.forward * Random.Range(-landingNoise, landingNoise);

        // ===== 随机飞行时间 =====
        float T = Random.Range(Tmin, Tmax);

        // ===== 有效重力 =====
        Vector3 g = Physics.gravity;

        // ===== 计算初速度 =====
        Vector3 v0 = (target - spawnPos - 0.5f * g * T * T) / T;


        // debug tag   for checking if land point is correct
        ballInstance.predictedLandingPoint = target;
        ballInstance.predictedT = T;
        ballInstance.ResetLandingCheck();

        
        ballInstance.rb.linearVelocity = v0;


        return ballInstance;
    }
}