/**
 * Created by mtambos on 08.07.16.
 */
public enum MixingStrategy {
    STACKED,
    SHUFFLED;

    public static MixingStrategy fromValue(int value) {
        switch(value) {
            case 0:
                return STACKED;
            case 1:
                return SHUFFLED;
            default:
                throw new RuntimeException("Unknown index:" + value);
        }
    }
}
