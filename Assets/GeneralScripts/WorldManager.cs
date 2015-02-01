using UnityEngine;
using System.Collections;

public class WorldManager : MonoBehaviour {

	public static int currentLevel = 0;

	private static int lastLevel;
	private static GameObject respawn;
	private static GameObject agent;

	void Start(){
		DontDestroyOnLoad(gameObject);
		agent = GameObject.FindWithTag("Player");
		DontDestroyOnLoad(agent);

		GameObject sensor = GameObject.FindWithTag("Sensor");
		DontDestroyOnLoad(sensor);
		lastLevel = Application.levelCount - 1;
	}

	public static void CompleteLevel(){
		if( currentLevel < lastLevel ){
			currentLevel ++;
			Application.LoadLevel(currentLevel);
		}
		else
			print ("You Win!");
	}

	private void OnLevelWasLoaded( int level ){
		respawn = GameObject.FindWithTag("Respawn");
		if( respawn != null )
			agent.transform.position = respawn.transform.position;

		SupervisedMotion sm = agent.GetComponent<SupervisedMotion>();
		sm.Start();

		if( currentLevel == lastLevel ){
			NetworkController n = agent.GetComponent<NetworkController>();
			n.recordMovement 	= false;
			n.TrainingPhase		= true;
		}
	}
}
