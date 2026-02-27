using UnityEngine;

public class CatchZone : MonoBehaviour
{
    public CarCatcherAgent agent;

    private void OnTriggerEnter(Collider other)
    {
        var ball = other.GetComponent<Ball>();
        if (ball != null)
        {
            agent.OnBallCaught();
        }
    }
}