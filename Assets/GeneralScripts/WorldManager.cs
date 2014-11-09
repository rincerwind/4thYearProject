using UnityEngine;
using System.Collections;

public class WorldManager : MonoBehaviour {

	public static int currentLevel = 0;
	private static int lastLevel;

	void Start(){
		DontDestroyOnLoad(gameObject);
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
}
