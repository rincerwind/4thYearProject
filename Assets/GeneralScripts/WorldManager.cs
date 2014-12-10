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
		/*NeuralNetwork[] nets = GetComponents<NeuralNetwork>();
		foreach (NeuralNetwork n in nets)
			DontDestroyOnLoad(n);*/
		DontDestroyOnLoad(agent);

		GameObject sensor = GameObject.FindWithTag("Sensor");
		DontDestroyOnLoad(sensor);
		lastLevel = Application.levelCount - 1;
	}

	public static void CompleteLevel(){
		if( currentLevel < lastLevel ){
			currentLevel ++;
			Application.LoadLevel(currentLevel);
			respawn = GameObject.FindWithTag("Respawn");
			agent.transform.position = respawn.transform.position;

			if( currentLevel == lastLevel ){
				NetworkController n = agent.GetComponent<NetworkController>();
				n.recordMovement 	= false;
				n.TrainingPhase		= true;
			}
		}
		else
			print ("You Win!");
	}
}
