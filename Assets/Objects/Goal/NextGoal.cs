using UnityEngine;
using System.Collections;

public class NextGoal : MonoBehaviour {
	public Transform[] goalPoints;
	private int currentGoal = 0;

	void Start(){
		transform.position = goalPoints[0].position;
	}

	public Transform getCurrentGoal(){
		return goalPoints[currentGoal];
	}

	public void goToNextGoal(){
		currentGoal++;
	}

	public bool isLastGoal(){
		return goalPoints.Length - 1 <= currentGoal;
	}
}
