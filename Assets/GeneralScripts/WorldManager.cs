using UnityEngine;
using System.Collections;

public class WorldManager : MonoBehaviour {

	public static int currentLevel = 0;
	private static int lastLevel;

	void Start(){
		NeuralNetwork[] nets = GetComponents<NeuralNetwork>();
		foreach (NeuralNetwork n in nets)
			DontDestroyOnLoad(n);

		DontDestroyOnLoad(gameObject);
		lastLevel = Application.levelCount - 1;
		print (lastLevel);
	}

	public static void CompleteLevel(){
		if( currentLevel < lastLevel ){
			currentLevel ++;
			Application.LoadLevel(currentLevel);
		}
		else
			print ("You Win!");
	}
}
